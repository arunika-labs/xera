"""
Automatic Flash Attention backend selection.

This module provides a single entry point, `auto_flash_attention`, that
picks a flash-attention implementation for the current JAX device:

    - TPU -> Splash Attention (Pallas TPU kernel, from jax.experimental.pallas)
    - GPU -> cuDNN fused attention (via jax.nn.dot_product_attention), which
             requires a sufficiently new GPU (Ampere/sm_80+), a compatible
             cuDNN version, and bf16/fp16 inputs -- cuDNN's fused kernel
             does not support fp32.
    - Anything else (CPU, etc.) -> unsupported; raises immediately.

There is no fallback kernel. If the current platform, requested dtype, or
requested feature (bias, local windowing) isn't supported by the backend
in play, this raises immediately with an explanation -- it never silently
substitutes a slower or different implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# cuDNN's fused attention kernel (as of the jax/jaxlib versions this module
# targets) only accepts these dtypes; fp32 in particular raises at runtime
# with a cuDNN-internal error that doesn't say "use bf16/fp16" anywhere in
# it. We check proactively so AutoFA's own error is the first thing the
# user sees.
_CUDNN_SUPPORTED_DTYPES = (jnp.bfloat16, jnp.float16)

# Splash attention's Pallas TPU kernel is written and tuned for bf16.
_SPLASH_SUPPORTED_DTYPES = (jnp.bfloat16,)


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
        return "additive bias is not supported by AutoFA's cuDNN path"
    if local_window_size is not None:
        return "local_window_size is not supported by AutoFA's cuDNN path"
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

def _splash_compatibility_issue(
    q: jax.Array, *, bias: jax.Array | None, local_window_size
) -> str | None:
    """
    Returns a reason string if splash attention is known not to support
    this call, or None if it looks compatible.
    """
    if q.dtype not in _SPLASH_SUPPORTED_DTYPES:
        supported = ", ".join(str(d) for d in _SPLASH_SUPPORTED_DTYPES)
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

_VALID_BACKENDS = ("auto", "cudnn", "splash")


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

    Picks a flash-attention implementation for the current JAX device:

        - TPU: Splash Attention (Pallas TPU kernel).
        - GPU: cuDNN fused attention.
        - Anything else: unsupported, raises `NotImplementedError`.

    There is no fallback kernel. If the platform, dtype, or requested
    feature (bias, local_window_size) isn't supported by the backend that
    would be used, this raises immediately (`NotImplementedError` for an
    unsupported platform, `ValueError` for an unsupported dtype/feature
    combination on an otherwise-supported platform).

    Args:
        q, k, v: Arrays of shape (batch, num_heads, seq_len, head_dim).
        causal: If True, apply a causal mask.
        scale: Softmax scale. Defaults to 1/sqrt(head_dim).
        bias: Optional additive attention bias. Not supported by either
            backend -- passing this always raises.
        local_window_size: Optional local attention window. Not supported
            by either backend -- passing this always raises.
        backend: One of "auto" (default), "cudnn", or "splash". Use a
            specific value to force that backend (raises if it's
            unavailable or doesn't support the requested dtype/features).

    Returns:
        Output array of shape (batch, num_heads, seq_len, head_dim).
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")

    platform = jax.devices()[0].platform

    if backend == "cudnn":
        issue = _cudnn_compatibility_issue(q, bias=bias, local_window_size=local_window_size)
        if issue is not None:
            raise ValueError(f"AutoFA: cannot use backend='cudnn': {issue}")
        return _flash_attention_cudnn(q, k, v, causal=causal, scale=scale)

    if backend == "splash":
        issue = _splash_compatibility_issue(q, bias=bias, local_window_size=local_window_size)
        if issue is not None:
            raise ValueError(f"AutoFA: cannot use backend='splash': {issue}")
        return _flash_attention_splash(q, k, v, causal=causal, scale=scale)

    # backend == "auto": dispatch by platform. No fallback -- unsupported
    # platform/dtype/feature combinations raise immediately.
    if platform == "tpu":
        issue = _splash_compatibility_issue(q, bias=bias, local_window_size=local_window_size)
        if issue is not None:
            raise ValueError(f"AutoFA: cannot run on TPU (backend='splash'): {issue}")
        return _flash_attention_splash(q, k, v, causal=causal, scale=scale)

    if platform == "gpu":
        issue = _cudnn_compatibility_issue(q, bias=bias, local_window_size=local_window_size)
        if issue is not None:
            raise ValueError(f"AutoFA: cannot run on GPU (backend='cudnn'): {issue}")
        return _flash_attention_cudnn(q, k, v, causal=causal, scale=scale)

    raise NotImplementedError(
        f"AutoFA has no supported backend for platform={platform!r}. "
        f"Supported platforms are 'tpu' (via splash attention) and 'gpu' "
        f"(via cuDNN fused attention)."
    )


__all__ = ["auto_flash_attention"]
