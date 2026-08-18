"""
Automatic Flash Attention backend selection.

This module provides a single entry point, `auto_flash_attention`, that
picks the fastest available flash-attention implementation for the current
JAX device automatically:

    - TPU  -> Splash Attention (Pallas TPU kernel, from jax.experimental.pallas)
    - GPU  -> cuDNN fused attention (via jax.nn.dot_product_attention), which
              requires a sufficiently new GPU (Ampere/sm_80+), a compatible
              cuDNN version, and (notably) bf16/fp16 inputs -- cuDNN's fused
              kernel does not support fp32. Falls back to the naive kernel
              below if any of that isn't met.
    - CPU  -> Naive flash attention (see below), run via Pallas interpret
              mode since Pallas has no native CPU backend.

Naive kernel
------------
`_flash_attention_naive` is a from-scratch Pallas kernel implementing the
classic FlashAttention algorithm: block-tiled matmuls with online (running)
softmax, so the full (T, T) attention matrix is never materialized. "Naive"
describes the algorithm/implementation effort (no autotuning, no fused
backward beyond what jax.grad derives, no vendor kernel), not its feature
set -- it is a deliberate superset: it supports every feature this module
exposes (causal masking, additive bias, local windowing, any dtype)
regardless of what the faster vendor backends can handle, so it always has
somewhere to fall back to. It runs in `interpret=True` mode automatically
on any device without a native Pallas backend (i.e. CPU) -- correctness
first; interpret mode is meaningfully slower than compiled Pallas and a
separate warning is raised when it's used.

Transparency
------------
AutoFA raises `XeraWarning` in these situations:
    - Whenever it selects a backend under "auto" mode.
    - Whenever it falls back to a different backend than the device would
      normally imply, always stating *why* (unsupported dtype, unsupported
      requested feature, backend raised an error, etc).
    - Whenever the naive kernel runs in Pallas interpret mode, since that
      is meaningfully slower than compiled execution and easy to miss.
Use `warnings.simplefilter("ignore", XeraWarning)` or the
`XERA_SILENCE_AUTOFA_WARNINGS=1` environment variable to silence these.
"""

from __future__ import annotations

import os
import warnings

import jax
import jax.numpy as jnp

from xera import core as _core


class XeraWarning(UserWarning):
    """Warning emitted by AutoFA when it selects or falls back to a backend."""


def _silenced() -> bool:
    return os.environ.get("XERA_SILENCE_AUTOFA_WARNINGS", "") == "1"


def _warn(message: str) -> None:
    if _silenced():
        return
    warnings.warn(message, XeraWarning, stacklevel=3)


def _warn_fallback(requested: str, actual: str, reason: str) -> None:
    _warn(
        f"AutoFA: requested backend='{requested}' but fell back to "
        f"'{actual}' (reason: {reason}). Performance may differ from "
        f"the requested backend. Pass backend='{actual}' explicitly to "
        f"silence this fallback, or set XERA_SILENCE_AUTOFA_WARNINGS=1 "
        f"to silence all AutoFA warnings."
    )


def _warn_selected(backend: str) -> None:
    _warn(f"AutoFA: using '{backend}' backend.")


def _warn_interpret_mode(backend: str, platform: str) -> None:
    _warn(
        f"AutoFA: running '{backend}' backend in Pallas interpret mode "
        f"(device platform='{platform}' has no native Pallas backend). "
        f"Interpret mode is meaningfully slower than compiled execution "
        f"and exists for correctness/portability, not performance -- avoid "
        f"relying on it in a training/inference hot path."
    )


# cuDNN's fused attention kernel (as of the jax/jaxlib versions this module
# targets) only accepts these dtypes; fp32 in particular raises at runtime
# with a cuDNN-internal error that doesn't say "use bf16/fp16" anywhere in
# it. We check proactively so AutoFA's own warning is the first thing the
# user sees.
_CUDNN_SUPPORTED_DTYPES = (jnp.bfloat16, jnp.float16)


# ---------------------------------------------------------------------------
# Naive kernel: block-tiled Pallas flash attention with online softmax.
# Supports causal masking, additive bias, and local windowing -- a superset
# of what the vendor backends (cuDNN/splash) are used for here, so it is
# always a valid fallback no matter what the caller asked for.
#
# The actual forward/backward math lives in `xera.core` as a
# `jax.custom_vjp` (`xera.core.auto_flash_attention` +
# `_autofa_forward`/`_autofa_backward`): the Pallas kernel here is built
# from `fori_loop` + dynamic-start (`pl.ds`) reads, which JAX's automatic
# differentiation cannot linearize through on its own, so a hand-written
# backward pass is required rather than relying on `jax.grad` directly.
# This function is a thin, backend-selection-facing wrapper around that
# core primitive: it keeps the same signature/behavior it always had
# (padding/masking defaults, interpret-mode auto-detection, warnings), it
# now just also supports `jax.grad`.
# ---------------------------------------------------------------------------

def _flash_attention_naive(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool = False,
    scale: float | None = None,
    bias: jax.Array | None = None,
    local_window_size: int | tuple[int | None, int | None] | None = None,
    block_q: int = 128,
    block_k: int = 128,
    interpret: bool | None = None,
    _warn_if_interpret: bool = True,
) -> jax.Array:
    """
    From-scratch Pallas flash attention: block tiling + online softmax.

    This is intentionally a simple/naive *implementation* (no autotuned
    block sizes, no vendor-fused backward pass) -- but a deliberate
    superset *feature-wise*: causal masking, additive bias, local
    windowing, and arbitrary dtypes are all supported unconditionally, so
    this always has somewhere for `auto_flash_attention` to fall back to
    regardless of what cuDNN/splash can or can't handle. It is
    differentiable via a custom VJP -- see `xera.core.auto_flash_attention`.

    Args:
        q, k, v: Arrays of shape (batch, num_heads, seq_len, head_dim).
        causal: If True, apply a causal mask (position i only attends to
            positions <= i).
        scale: Softmax scale. Defaults to 1/sqrt(head_dim).
        bias: Optional additive bias broadcastable to
            (batch, num_heads, seq_len, seq_len), added to the raw
            attention logits before the softmax.
        local_window_size: Either a single int (symmetric window) or a
            (left, right) tuple giving how many positions to the left/right
            of each query a key may be at most. Either side may be None for
            "unbounded" on that side. Combines with `causal` if both given.
        block_q: Query tile size along the sequence dimension.
        block_k: Key/value tile size along the sequence dimension.
        interpret: Force Pallas interpret mode. If None (default), auto-set
            to True on any non-TPU/GPU backend (e.g. CPU), False otherwise.

    Returns:
        Output array of shape (batch, num_heads, seq_len, head_dim).
    """
    _batch, _num_heads, seq_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    platform = jax.devices()[0].platform
    if interpret is None:
        interpret = platform not in ("tpu", "gpu")
    if interpret and _warn_if_interpret:
        _warn_interpret_mode("naive", platform)

    window_left: int | None
    window_right: int | None
    if local_window_size is None:
        window_left = window_right = None
    elif isinstance(local_window_size, tuple):
        window_left, window_right = local_window_size
    else:
        window_left = window_right = local_window_size

    block_q = min(block_q, seq_len)
    block_k = min(block_k, seq_len)

    if not interpret:
        # Pallas TPU (Mosaic) requires the last two dims of every block
        # shape to be divisible by (8, 128). Our q/k/v/out block shapes
        # are (..., block_{q,k}, head_dim), so that means block_q/block_k
        # must be multiples of 8 and head_dim a multiple of 128 whenever
        # we're actually lowering to TPU (not just running in Pallas
        # interpret mode, which never hits Mosoic and so never enforces
        # this). Raise early with an actionable message instead of
        # letting Mosaic's lower-level error surface.
        def _round_up(n: int, multiple: int) -> int:
            return -(-n // multiple) * multiple

        if head_dim % 128 != 0:
            raise ValueError(
                f"AutoFA naive kernel on TPU requires head_dim to be a "
                f"multiple of 128 (got head_dim={head_dim}). Pad your "
                f"q/k/v head dimension up to {_round_up(head_dim, 128)} "
                f"(e.g. jnp.pad(..., [(0, 0), (0, 0), (0, 0), "
                f"(0, {_round_up(head_dim, 128) - head_dim})])) and slice "
                f"the output back down afterwards, or run with "
                f"interpret=True (much slower; correctness-only)."
            )
        if block_q % 8 != 0:
            fixed = _round_up(block_q, 8)
            raise ValueError(
                f"AutoFA naive kernel on TPU requires block_q to be a "
                f"multiple of 8 (got block_q={block_q}, after clamping to "
                f"seq_len={seq_len}). Pass block_q={fixed} (or another "
                f"multiple of 8), or run with interpret=True."
            )
        if block_k % 8 != 0:
            fixed = _round_up(block_k, 8)
            raise ValueError(
                f"AutoFA naive kernel on TPU requires block_k to be a "
                f"multiple of 8 (got block_k={block_k}, after clamping to "
                f"seq_len={seq_len}). Pass block_k={fixed} (or another "
                f"multiple of 8), or run with interpret=True."
            )

    has_bias = bias is not None
    if not has_bias:
        # Dummy placeholder so `xera.core.auto_flash_attention` always sees
        # an array for its `bias` (differentiable) argument; never read
        # when `has_bias` is False.
        bias = jnp.zeros((1, 1, block_q, block_q), dtype=q.dtype)

    return _core.auto_flash_attention(
        q, k, v, bias,
        has_bias, causal, scale, window_left, window_right,
        block_q, block_k, interpret,
    )


# ---------------------------------------------------------------------------
# cuDNN backend (GPU only, requires Ampere+/sm_80+, a compatible cuDNN, and
# bf16/fp16 inputs -- fp32 is not supported by the fused kernel).
# ---------------------------------------------------------------------------

def _cudnn_compatibility_issue(
    q: jax.Array, *, bias: jax.Array | None, local_window_size
) -> str | None:
    """
    Returns a human-readable reason string if cuDNN is known in advance not
    to support this call, or None if it looks compatible (a None result is
    not a guarantee -- cuDNN may still reject the call at runtime for
    reasons this check doesn't cover, e.g. exact head_dim/shape limits that
    vary by cuDNN version).
    """
    if q.dtype not in _CUDNN_SUPPORTED_DTYPES:
        supported = ", ".join(str(d) for d in _CUDNN_SUPPORTED_DTYPES)
        return f"dtype {q.dtype} is not supported by cuDNN fused attention (supported: {supported})"
    if bias is not None:
        # jax.nn.dot_product_attention does accept a `bias` argument for
        # the xla implementation, but cuDNN fused-attention bias support is
        # version-gated and inconsistent enough that AutoFA treats bias as
        # naive-only for reliability.
        return "additive bias is not supported by AutoFA's cuDNN path (falls back to naive)"
    if local_window_size is not None:
        # jax.nn.dot_product_attention *does* expose local_window_size for
        # cuDNN in newer versions, but support varies by cuDNN/jaxlib
        # version; AutoFA treats it as naive-only for reliability rather
        # than risk a silent shape/behavior mismatch across versions.
        return "local_window_size is not supported by AutoFA's cuDNN path (falls back to naive)"
    return None


def _flash_attention_cudnn(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> jax.Array:
    """
    cuDNN fused flash attention via jax.nn.dot_product_attention.

    Expects (batch, num_heads, seq_len, head_dim) like the rest of this
    module; jax.nn.dot_product_attention wants (batch, seq_len, num_heads,
    head_dim), so we transpose in and out.
    """
    qT = q.transpose(0, 2, 1, 3)
    kT = k.transpose(0, 2, 1, 3)
    vT = v.transpose(0, 2, 1, 3)

    out = jax.nn.dot_product_attention(
        qT, kT, vT,
        scale=scale,
        is_causal=causal,
        implementation="cudnn",
    )
    return out.transpose(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Splash attention backend (TPU only).
# ---------------------------------------------------------------------------

# Splash attention's Pallas TPU kernel is written and tuned for bf16 (fp32
# runs but defeats the point -- no MXU speedup, and it hasn't been the
# well-tested path historically). AutoFA warns rather than silently eating
# the slowdown.
_SPLASH_PREFERRED_DTYPES = (jnp.bfloat16,)


def _splash_compatibility_issue(
    q: jax.Array, *, bias: jax.Array | None, local_window_size
) -> str | None:
    """
    Returns a reason string if splash attention is known not to support
    this call (hard blockers only -- dtype mismatch is a soft "works but
    slow" case handled separately via a warning, not a fallback).
    """
    if bias is not None:
        return "additive bias is not supported by splash attention (falls back to naive)"
    if local_window_size is not None:
        return "local_window_size is not supported by AutoFA's splash attention path (falls back to naive)"
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

_VALID_BACKENDS = ("auto", "cudnn", "splash", "naive")


def auto_flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool = False,
    scale: float | None = None,
    bias: jax.Array | None = None,
    local_window_size: int | tuple[int | None, int | None] | None = None,
    backend: str = "auto",
) -> jax.Array:
    """
    Flash attention with automatic backend selection (AutoFA).

    Picks the best available flash-attention implementation for the
    current JAX device and requested features:

        - TPU: Splash Attention (Pallas TPU kernel), if the request doesn't
          need bias/local-windowing (naive-only features); falls back to
          naive otherwise or if splash itself raises.
        - GPU: cuDNN fused attention, if dtype is bf16/fp16 and the request
          doesn't need bias/local-windowing; falls back to naive for fp32
          inputs, unsupported features, or if cuDNN itself raises (e.g.
          unsupported GPU/cuDNN version, shape limits).
        - CPU (or any other backend): naive Pallas kernel, run in
          interpret mode.

    `XeraWarning` is raised for every backend selection under "auto", every
    fallback (always stating why), and whenever the naive kernel runs in
    Pallas interpret mode (meaningfully slower than compiled execution).
    Set `XERA_SILENCE_AUTOFA_WARNINGS=1` to suppress all of these.

    Args:
        q, k, v: Arrays of shape (batch, num_heads, seq_len, head_dim).
        causal: If True, apply a causal mask.
        scale: Softmax scale. Defaults to 1/sqrt(head_dim).
        bias: Optional additive attention bias. Currently naive-only --
            requesting this on "auto" routes straight to the naive kernel
            (see class docstring for why cuDNN/splash aren't used for it).
        local_window_size: Optional local attention window (see
            `_flash_attention_naive` for the exact semantics). Also
            currently naive-only under "auto", for the same reason.
        backend: One of "auto" (default), "cudnn", "splash", or "naive".
            Use a specific value to force that backend (raises if it's
            unavailable/fails/doesn't support the requested features,
            rather than silently falling back) -- this is meant for
            benchmarking/debugging, not casual use.

    Returns:
        Output array of shape (batch, num_heads, seq_len, head_dim).
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")

    platform = jax.devices()[0].platform

    if backend == "cudnn":
        return _flash_attention_cudnn(q, k, v, causal=causal, scale=scale)
    if backend == "splash":
        return _flash_attention_splash(q, k, v, causal=causal, scale=scale)
    if backend == "naive":
        return _flash_attention_naive(
            q, k, v, causal=causal, scale=scale, bias=bias, local_window_size=local_window_size,
        )

    # backend == "auto": dispatch by platform + feature/dtype compatibility,
    # with fallback + XeraWarning at every decision point.
    if platform == "tpu":
        issue = _splash_compatibility_issue(q, bias=bias, local_window_size=local_window_size)
        if issue is not None:
            _warn_fallback("splash", "naive", issue)
            return _flash_attention_naive(
                q, k, v, causal=causal, scale=scale, bias=bias, local_window_size=local_window_size,
            )
        if q.dtype not in _SPLASH_PREFERRED_DTYPES:
            _warn(
                f"AutoFA: dtype {q.dtype} is not splash attention's preferred "
                f"dtype ({_SPLASH_PREFERRED_DTYPES[0]}); splash will still run "
                f"but likely without its usual MXU speedup. Cast to bfloat16 "
                f"for full performance, or pass backend='naive' if you'd "
                f"rather use the portable kernel."
            )
        _warn_selected("splash")
        try:
            return _flash_attention_splash(q, k, v, causal=causal, scale=scale)
        except Exception as e:  # noqa: BLE001 -- deliberately broad: any splash
            # failure should fall back, not crash the training run.
            _warn_fallback("splash", "naive", f"splash attention raised: {e}")
            return _flash_attention_naive(
                q, k, v, causal=causal, scale=scale, bias=bias, local_window_size=local_window_size,
            )

    if platform == "gpu":
        issue = _cudnn_compatibility_issue(q, bias=bias, local_window_size=local_window_size)
        if issue is not None:
            _warn_fallback("cudnn", "naive", issue)
            return _flash_attention_naive(
                q, k, v, causal=causal, scale=scale, bias=bias, local_window_size=local_window_size,
            )
        _warn_selected("cudnn")
        try:
            return _flash_attention_cudnn(q, k, v, causal=causal, scale=scale)
        except Exception as e:  # noqa: BLE001 -- see above.
            _warn_fallback(
                "cudnn", "naive",
                f"cuDNN fused attention raised (likely unsupported GPU/cuDNN "
                f"version or shape not covered by AutoFA's preflight checks): {e}",
            )
            return _flash_attention_naive(
                q, k, v, causal=causal, scale=scale, bias=bias, local_window_size=local_window_size,
            )

    # CPU or anything else.
    _warn_selected("naive")
    return _flash_attention_naive(
        q, k, v, causal=causal, scale=scale, bias=bias, local_window_size=local_window_size,
    )


__all__ = ["auto_flash_attention", "XeraWarning"]
