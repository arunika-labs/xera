"""
Scaled dot-product flash attention, with automatic backend selection.

This module provides a single entry point, `sdpa_flash`, that picks a
flash-attention implementation for the current JAX device:

    - TPU -> Splash Attention (Pallas TPU kernel, from jax.experimental.pallas),
             if the request satisfies its known constraints (bf16, head_dim a
             multiple of 128, no bias/local_window_size -- splash's Pallas
             kernel has no additive-bias input and no windowed masking).
    - GPU -> cuDNN fused attention (via jax.nn.dot_product_attention), which
             requires a sufficiently new GPU (Ampere/sm_80+, checked via the
             device's `compute_capability`), a compatible cuDNN version, and
             bf16/fp16 inputs -- cuDNN's fused kernel does not support fp32.
             Additive bias and local_window_size are both supported here:
             `jax.nn.dot_product_attention`'s `cudnn` implementation accepts
             a `bias` array (added to logits pre-softmax) and a
             `local_window_size` argument (lowered to cuDNN's native sliding
             window), and this backend forwards both through.
             (The public `jax.nn.dot_product_attention` wrapper this backend
             calls does not expose cuDNN's fp8 path, so fp8 is not a
             supported dtype here even though cuDNN itself can do fp8.)
    - Anything else (CPU, etc.), or a GPU/TPU call that the vendor backend
      above can't serve (unsupported dtype/hardware, or -- on TPU only --
      a requested feature like bias/local windowing that splash doesn't
      support) -> `jax_flash_attention` (see `xera.loom.jax_flash_attention`),
      a pure-jnp tiled attention with online softmax. Dtype-, device-, and
      feature-agnostic by construction, so it always works as a fallback.

By default, `sdpa_flash` never prints anything -- whether `backend="auto"`
silently used the portable "jax" backend or the vendor kernel is not
reported unless the caller passes `verbose=True`, in which case a single
`XeraInfo: ...` line explains *why* dispatch fell back to "jax" (nothing
is printed when the vendor backend is used successfully, or when
`backend=` is forced explicitly).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .jax_flash_attention import jax_flash_attention


# cuDNN's fused attention kernel (as of the jax/jaxlib versions this module
# targets) only accepts these dtypes; fp32 in particular raises at runtime
# with a cuDNN-internal error that doesn't say "use bf16/fp16" anywhere in
# it. We check proactively so AutoFA's own error is the first thing the
# user sees.
#
# fp8 is deliberately excluded: cuDNN's fused kernel supports fp8, but only
# through the internal `fp8_params`/`use_fp8` arguments of
# `jax._src.cudnn.fused_attention_stablehlo.dot_product_attention` -- the
# public `jax.nn.dot_product_attention` wrapper this backend calls does not
# expose them, so fp8 isn't reachable through this code path regardless of
# hardware/cuDNN version.
_CUDNN_SUPPORTED_DTYPES = (jnp.bfloat16, jnp.float16)

# cuDNN fused attention requires Ampere or newer (sm_80+).
_CUDNN_MIN_COMPUTE_CAPABILITY = (8, 0)

# Splash attention's Pallas TPU kernel is written and tuned for bf16.
_SPLASH_SUPPORTED_DTYPES = (jnp.bfloat16,)

# Splash attention's kernel requires head_dim to be a multiple of 128 (see
# jax-ml/jax#26433) -- e.g. DeepSeek's head_dim=192 is not supported.
_SPLASH_HEAD_DIM_MULTIPLE = 128


def _info(reason: str, *, verbose: bool) -> None:
    if verbose:
        print(f"XeraInfo: sdpa_flash falling back to backend='jax', because {reason}.")


def _parse_compute_capability(device) -> tuple[int, int] | None:
    """
    Best-effort parse of a GPU device's `compute_capability` (e.g. "8.0"
    for an A100) into a `(major, minor)` tuple.

    Returns None if the device doesn't expose a `compute_capability`
    attribute, or its value isn't in the expected "MAJOR.MINOR" string
    form -- callers must treat None as "unknown", not as "compatible".
    """
    compute_capability = getattr(device, "compute_capability", None)
    if compute_capability is None:
        return None
    try:
        major_str, minor_str = str(compute_capability).split(".")
        return int(major_str), int(minor_str)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# cuDNN backend (GPU only, requires Ampere+/sm_80+, a compatible cuDNN, and
# bf16/fp16 inputs -- fp32 is not supported by the fused kernel).
# ---------------------------------------------------------------------------

def _cudnn_compatibility_issue(
    q: jax.Array, *, device, bias: jax.Array | None, local_window_size
) -> str | None:
    """
    Returns a human-readable reason string if cuDNN is known in advance not
    to support this call, or None if it looks compatible (a None result is
    not a guarantee -- cuDNN may still reject the call at runtime for
    reasons this check doesn't cover, e.g. exact head_dim/shape limits that
    vary by cuDNN version).

    `bias` and `local_window_size` are accepted parameters here (not
    rejected): `jax.nn.dot_product_attention`'s `cudnn` implementation
    forwards both straight into cuDNN's fused kernel (an additive `bias`
    argument, and `local_window_size` lowered to cuDNN's native sliding
    window support). They're only kept as parameters on this function so
    its signature matches `_splash_compatibility_issue` for a uniform call
    site in the dispatcher below -- splash's kernel is the one that can't
    serve them.
    """
    compute_capability = _parse_compute_capability(device)
    min_major, min_minor = _CUDNN_MIN_COMPUTE_CAPABILITY
    if compute_capability is None:
        return (
            f"this GPU's compute capability could not be determined, and cuDNN "
            f"fused attention requires sm_{min_major}{min_minor}+ (Ampere or newer)"
        )
    if compute_capability < _CUDNN_MIN_COMPUTE_CAPABILITY:
        major, minor = compute_capability
        return (
            f"GPU compute capability sm_{major}{minor} is below the "
            f"sm_{min_major}{min_minor} (Ampere) minimum required by cuDNN fused attention"
        )
    if q.dtype not in _CUDNN_SUPPORTED_DTYPES:
        supported = ", ".join(jnp.dtype(d).name for d in _CUDNN_SUPPORTED_DTYPES)
        return f"dtype {q.dtype} is not supported by cuDNN fused attention (supported: {supported})"
    return None


def _flash_attention_cudnn(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool = False,
    scale: float | None = None,
    bias: jax.Array | None = None,
    local_window_size: int | tuple[int | None, int | None] | None = None,
) -> jax.Array:
    """
    cuDNN fused flash attention via jax.nn.dot_product_attention.

    Expects (batch, num_heads, seq_len, head_dim) like the rest of this
    module; jax.nn.dot_product_attention wants (batch, seq_len, num_heads,
    head_dim), so we transpose in and out.

    `bias` -- if given -- follows the same (batch, num_heads, seq_len,
    seq_len) convention as the rest of this module (matching the "jax"
    backend's `bias` argument), and is transposed to
    (batch, seq_len, num_heads, seq_len) to match q/k/v's transposed
    layout; jax.nn.dot_product_attention broadcasts it against logits of
    shape (B, T, N, S) from there. `local_window_size` is passed straight
    through to jax.nn.dot_product_attention, which lowers it to cuDNN's
    native sliding-window attention.
    """
    qT = q.transpose(0, 2, 1, 3)
    kT = k.transpose(0, 2, 1, 3)
    vT = v.transpose(0, 2, 1, 3)
    biasT = bias.transpose(0, 2, 1, 3) if bias is not None else None

    out = jax.nn.dot_product_attention(
        qT, kT, vT,
        bias=biasT,
        scale=scale,
        is_causal=causal,
        local_window_size=local_window_size,
        implementation="cudnn",
    )
    return out.transpose(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Splash attention backend (TPU only).
# ---------------------------------------------------------------------------

def _splash_compatibility_issue(
    q: jax.Array, *, bias: jax.Array | None, local_window_size
) -> str | None:
    """
    Returns a reason string if splash attention is known not to support
    this call, or None if it looks compatible.

    Unlike cuDNN, splash's Pallas TPU kernel genuinely has no additive-bias
    input and no windowed-masking support, so `bias`/`local_window_size`
    are hard rejections here, not just unforwarded arguments.
    """
    head_dim = q.shape[-1]
    if head_dim % _SPLASH_HEAD_DIM_MULTIPLE != 0:
        return (
            f"head_dim {head_dim} is not a multiple of {_SPLASH_HEAD_DIM_MULTIPLE}, "
            "required by splash attention's Pallas TPU kernel"
        )
    if q.dtype not in _SPLASH_SUPPORTED_DTYPES:
        supported = ", ".join(jnp.dtype(d).name for d in _SPLASH_SUPPORTED_DTYPES)
        return f"dtype {q.dtype} is not supported by splash attention (supported: {supported})"
    if bias is not None:
        return "additive bias is not supported by splash attention"
    if local_window_size is not None:
        return "local_window_size is not supported by AutoFA's splash attention path"
    return None


def _flash_attention_splash(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> jax.Array:
    """
    TPU Splash Attention, via jax.experimental.pallas.ops.tpu.splash_attention.

    Splash attention expects (num_heads, seq_len, head_dim) per batch
    element and is vmapped here over the batch dimension.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sak
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sam

    _, num_heads, seq_len, head_dim = q.shape
    if scale is not None:
        q = q * (scale * (head_dim ** 0.5))  # splash applies its own 1/sqrt(head_dim)

    mask = sam.CausalMask((seq_len, seq_len)) if causal else sam.FullMask((seq_len, seq_len))
    multi_head_mask = sam.MultiHeadMask(masks=[mask] * num_heads)
    block_sizes = sak.BlockSizes.get_default()
    kernel = sak.make_splash_mha(mask=multi_head_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes)

    def per_example(q_, k_, v_):
        return kernel(q_, k_, v_)

    return jax.vmap(per_example)(q, k, v)


# ---------------------------------------------------------------------------
# Public dispatcher.
# ---------------------------------------------------------------------------

_VALID_BACKENDS = ("auto", "cudnn", "splash", "jax")

# Default tile sizes jax_flash_attention uses when reached via auto-dispatch/vendor-fallback
# (not user-configurable through this entry point -- call
# `xera.loom.jax_flash_attention.jax_flash_attention` directly for that).
_JAX_FLASH_ATTENTION_BLOCK_Q = 128
_JAX_FLASH_ATTENTION_BLOCK_K = 128


def _flash_attention_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool = False,
    scale: float | None = None,
    bias: jax.Array | None = None,
    local_window_size: int | tuple[int | None, int | None] | None = None,
) -> jax.Array:
    """`sdpa_flash`'s portable-kernel path -- see `xera.loom.jax_flash_attention`."""
    if local_window_size is None:
        window_left = window_right = None
    elif isinstance(local_window_size, tuple):
        window_left, window_right = local_window_size
    else:
        window_left = window_right = local_window_size

    return jax_flash_attention(
        q, k, v, bias,
        causal, scale, window_left, window_right,
        _JAX_FLASH_ATTENTION_BLOCK_Q, _JAX_FLASH_ATTENTION_BLOCK_K,
    )


def sdpa_flash(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool = False,
    scale: float | None = None,
    bias: jax.Array | None = None,
    local_window_size: int | tuple[int | None, int | None] | None = None,
    backend: str = "auto",
    verbose: bool = False,
) -> jax.Array:
    """
    Scaled dot-product flash attention, with automatic backend selection.

    By default (`backend="auto"`), picks a flash-attention implementation
    for the current JAX device:

        - TPU: Splash Attention (Pallas TPU kernel), if dtype/head_dim/
          features are supported (bf16, head_dim a multiple of 128, no
          bias/local_window_size -- splash's kernel supports neither);
          falls back to the portable `"jax"` backend otherwise.
        - GPU: cuDNN fused attention, if the GPU is Ampere/sm_80+ (checked
          via the device's `compute_capability`) and dtype is supported
          (bf16/fp16); falls back to the portable `"jax"` backend
          otherwise. Unlike splash, cuDNN supports both `bias` and
          `local_window_size` natively, so requesting them doesn't trigger
          a fallback on GPU.
        - Anything else (CPU, etc.): the portable `"jax"` backend, always
          -- it's the only option there, not a "fallback" from anything.

    The `"jax"` backend (`xera.loom.jax_flash_attention`) is a pure-jnp
    tiled attention with online softmax: dtype-, device-, and
    feature-agnostic, so it always works regardless of platform, dtype,
    or whether bias/local_window_size were requested.

    By default (`verbose=False`), `sdpa_flash` never prints anything --
    whether `backend="auto"` used the vendor kernel or silently fell back
    to `"jax"` is not reported, so callers get the same output either way
    with nothing written to stdout. Pass `verbose=True` to have a single
    line printed whenever `backend="auto"` falls back to `"jax"`,
    explaining why:

        XeraInfo: sdpa_flash falling back to backend='jax', because <reason>.

    This is a plain `print()`, not a `warnings.warn` -- it's routine,
    expected behavior (not a problem to flag), so it doesn't carry a
    warning's weight. It never fires when cuDNN/splash is used
    successfully, and never fires under an explicitly forced `backend=`
    (forcing a backend either works or raises -- there's no fallback to
    explain).

    Args:
        q, k, v: Arrays of shape (batch, num_heads, seq_len, head_dim).
        causal: If True, apply a causal mask.
        scale: Softmax scale. Defaults to 1/sqrt(head_dim).
        bias: Optional additive attention bias, added to logits before the
            softmax. Shape (batch, num_heads, seq_len, seq_len) (or
            broadcastable to it), matching q/k/v's (batch, num_heads,
            seq_len, head_dim) convention. Supported by the `"jax"` and
            `"cudnn"` backends; requesting this with `backend="auto"` on
            TPU routes to `"jax"`, since splash's kernel has no bias input.
        local_window_size: Optional local attention window. Supported by
            the `"jax"` and `"cudnn"` backends, same as `bias` -- routes to
            `"jax"` on TPU under `backend="auto"`, since splash doesn't
            support it either.
        backend: `"auto"` (default) for automatic dispatch. Set to
            `"jax"`, `"cudnn"`, or `"splash"` to force that backend
            (raises if it's unavailable or doesn't support the requested
            dtype/features -- forcing a backend means "use exactly this,
            or fail", no fallback).
        verbose: If True, print a single line when `backend="auto"` falls
            back to `"jax"`, explaining why. Defaults to False, in which
            case dispatch is completely silent regardless of outcome.

    Returns:
        Output array of shape (batch, num_heads, seq_len, head_dim).
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")

    device = jax.devices()[0]
    platform = device.platform

    if backend == "cudnn":
        issue = _cudnn_compatibility_issue(q, device=device, bias=bias, local_window_size=local_window_size)
        if issue is not None:
            raise ValueError(f"AutoFA: cannot use backend='cudnn': {issue}")
        return _flash_attention_cudnn(
            q, k, v, causal=causal, scale=scale, bias=bias, local_window_size=local_window_size,
        )

    if backend == "splash":
        issue = _splash_compatibility_issue(q, bias=bias, local_window_size=local_window_size)
        if issue is not None:
            raise ValueError(f"AutoFA: cannot use backend='splash': {issue}")
        return _flash_attention_splash(q, k, v, causal=causal, scale=scale)

    if backend == "jax":
        return _flash_attention_jax(
            q, k, v, causal=causal, scale=scale, bias=bias, local_window_size=local_window_size,
        )

    # backend == "auto": dispatch by platform, falling back to the
    # portable "jax" backend whenever the vendor backend for that
    # platform can't serve the request. CPU (or anything else) always
    # uses "jax" silently -- it's the only backend there.
    if platform == "tpu":
        issue = _splash_compatibility_issue(q, bias=bias, local_window_size=local_window_size)
        if issue is not None:
            _info(issue, verbose=verbose)
            return _flash_attention_jax(
                q, k, v, causal=causal, scale=scale, bias=bias, local_window_size=local_window_size,
            )
        return _flash_attention_splash(q, k, v, causal=causal, scale=scale)

    if platform == "gpu":
        issue = _cudnn_compatibility_issue(q, device=device, bias=bias, local_window_size=local_window_size)
        if issue is not None:
            _info(issue, verbose=verbose)
            return _flash_attention_jax(
                q, k, v, causal=causal, scale=scale, bias=bias, local_window_size=local_window_size,
            )
        return _flash_attention_cudnn(
            q, k, v, causal=causal, scale=scale, bias=bias, local_window_size=local_window_size,
        )

    # CPU or anything else: "jax" is simply the only backend -- no
    # fallback happened, so nothing is printed regardless of `verbose`.
    return _flash_attention_jax(
        q, k, v, causal=causal, scale=scale, bias=bias, local_window_size=local_window_size,
    )


__all__ = ["sdpa_flash"]
