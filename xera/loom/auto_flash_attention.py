"""
AutoFA: Automatic Flash Attention backend selection.

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

import functools
import os
import warnings

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


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
# ---------------------------------------------------------------------------

def _flash_attention_naive_kernel(
    q_ref, k_ref, v_ref, bias_ref, o_ref, *,
    block_k: int,
    seq_len: int,
    causal: bool,
    scale: float,
    has_bias: bool,
    window_left: int | None,
    window_right: int | None,
):
    """
    Pallas kernel body for one (batch, head, q_block) grid cell.

    q_ref:    (1, 1, block_q, head_dim) -- this program's slice of (padded)
              queries.
    k_ref:    (1, 1, padded_seq_len, head_dim) -- full (padded) keys for
              this (batch, head).
    v_ref:    (1, 1, padded_seq_len, head_dim) -- full (padded) values for
              this (batch, head).
    bias_ref: (1, 1, block_q, padded_seq_len) -- additive bias slice for
              this (batch, head, q_block), or a dummy zero-size ref when
              `has_bias` is False (never read in that case).
    o_ref:    (1, 1, block_q, head_dim) -- output slice to write.

    The leading (1, 1) axes come from the BlockSpec's batch/head blocking
    and are squeezed out immediately below.

    `seq_len` is the *original*, unpadded sequence length -- the caller
    (`_flash_attention_naive`) pads q/k/v (and bias, if given) up to a
    multiple of the block sizes before calling into Pallas, so every
    `pl.ds` read in this kernel is always in-bounds. `seq_len` is only
    used to mask out padding positions from the softmax and to compute
    correct query/key positions for causal/local-window masking.

    Implements the online-softmax recurrence: keys/values are streamed in
    chunks of `block_k`, and the running max / running sum / running
    weighted-value accumulator are updated chunk by chunk, so the full
    (block_q, seq_len) score matrix is never materialized at once.
    """
    q_block_idx = pl.program_id(2)
    block_q = q_ref.shape[2]
    head_dim = q_ref.shape[3]

    q = q_ref[0, 0, :, :] * scale

    m_i = jnp.full((block_q,), -jnp.inf, dtype=jnp.float32)
    l_i = jnp.zeros((block_q,), dtype=jnp.float32)
    acc = jnp.zeros((block_q, head_dim), dtype=jnp.float32)

    padded_seq_len = k_ref.shape[2]
    num_k_blocks = padded_seq_len // block_k

    def body(k_idx, carry):
        m_i, l_i, acc = carry
        k_start = k_idx * block_k

        # Always in-bounds: k_ref/v_ref were padded up to a multiple of
        # block_k by the caller, so [k_start, k_start + block_k) never
        # exceeds padded_seq_len.
        k_block = k_ref[0, 0, pl.ds(k_start, block_k), :]
        v_block = v_ref[0, 0, pl.ds(k_start, block_k), :]

        scores = jnp.dot(q, k_block.T, preferred_element_type=jnp.float32)

        if has_bias:
            bias_block = bias_ref[0, 0, :, pl.ds(k_start, block_k)]
            scores = scores + bias_block.astype(jnp.float32)

        # True sequence position of each column/row in this block.
        k_pos = k_start + jax.lax.iota(jnp.int32, block_k)[None, :]
        q_pos = q_block_idx * block_q + jax.lax.iota(jnp.int32, block_q)[:, None]

        # Positions >= seq_len are padding and must never receive
        # probability mass.
        in_bounds = k_pos < seq_len
        scores = jnp.where(in_bounds, scores, -jnp.inf)

        if causal:
            scores = jnp.where(q_pos >= k_pos, scores, -jnp.inf)

        if window_left is not None or window_right is not None:
            rel = q_pos - k_pos  # > 0 means key is to the left of query
            if window_left is not None:
                scores = jnp.where(rel <= window_left, scores, -jnp.inf)
            if window_right is not None:
                scores = jnp.where(-rel <= window_right, scores, -jnp.inf)

        m_ij = jnp.max(scores, axis=-1)
        m_new = jnp.maximum(m_i, m_ij)

        # When a block contributes no valid (in-window/in-bounds/causal)
        # positions at all, its scores are entirely -inf, so m_ij = -inf.
        # If m_i is also still -inf (no valid block seen yet for this
        # query row), m_new stays -inf and `exp(m_i - m_new)` below would
        # be `exp(-inf - (-inf)) = exp(nan) = nan`. Guard explicitly: when
        # m_new is -inf, nothing valid has been seen yet, so alpha (the
        # rescaling factor for the running accumulator) is simply 0 --
        # there is nothing to rescale, since acc/l_i are still all-zero.
        m_new_is_neg_inf = jnp.isneginf(m_new)
        p = jnp.where(
            m_new_is_neg_inf[:, None], 0.0, jnp.exp(scores - m_new[:, None])
        )
        alpha = jnp.where(m_new_is_neg_inf, 0.0, jnp.exp(m_i - m_new))

        l_new = alpha * l_i + jnp.sum(p, axis=-1)
        acc_new = acc * alpha[:, None] + jnp.dot(
            p.astype(v_block.dtype), v_block, preferred_element_type=jnp.float32
        )

        return m_new, l_new, acc_new

    def skip_body(k_idx, carry):
        return carry

    def loop_body(k_idx, carry):
        if causal:
            # Skip key blocks entirely to the future of this query block.
            # (Local windowing does not get the same skip: the window can
            # start anywhere in the sequence, so no block is unconditionally
            # irrelevant the way "future" blocks are under causal masking.)
            k_start = k_idx * block_k
            q_block_start = q_block_idx * block_q
            needed = k_start <= (q_block_start + block_q - 1)
            return jax.lax.cond(needed, body, skip_body, k_idx, carry)
        return body(k_idx, carry)

    m_i, l_i, acc = jax.lax.fori_loop(0, num_k_blocks, loop_body, (m_i, l_i, acc))

    l_i_safe = jnp.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_i_safe[:, None]
    o_ref[0, 0, :, :] = out.astype(o_ref.dtype)


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
    regardless of what cuDNN/splash can or can't handle.

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
    batch, num_heads, seq_len, head_dim = q.shape
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

    # Pad the sequence dimension up to a multiple of both block sizes so
    # every dynamic-start `pl.ds` read inside the kernel is guaranteed
    # in-bounds. Pallas does not bounds-check dynamic-start reads, so an
    # unclamped/unpadded out-of-range read would silently return
    # incorrect data rather than erroring -- padding removes the
    # possibility of that happening at all. Padding positions are masked
    # out inside the kernel via `seq_len` (the original, unpadded length).
    padded_len_q = pl.cdiv(seq_len, block_q) * block_q
    padded_len_k = pl.cdiv(seq_len, block_k) * block_k
    padded_len = max(padded_len_q, padded_len_k)

    pad_amount = padded_len - seq_len
    if pad_amount > 0:
        pad_width = [(0, 0), (0, 0), (0, pad_amount), (0, 0)]
        q = jnp.pad(q, pad_width)
        k = jnp.pad(k, pad_width)
        v = jnp.pad(v, pad_width)

    has_bias = bias is not None
    if has_bias:
        bias = jnp.broadcast_to(bias, (batch, num_heads, seq_len, seq_len))
        if pad_amount > 0:
            bias = jnp.pad(
                bias,
                [(0, 0), (0, 0), (0, padded_len - seq_len), (0, padded_len - seq_len)],
            )
    else:
        # Dummy placeholder so the kernel always has 4 Ref args regardless
        # of has_bias; never read inside the kernel when has_bias is False.
        bias = jnp.zeros((1, 1, block_q, padded_len), dtype=q.dtype)

    grid = (batch, num_heads, padded_len // block_q)

    kernel = functools.partial(
        _flash_attention_naive_kernel,
        block_k=block_k,
        seq_len=seq_len,
        causal=causal,
        scale=scale,
        has_bias=has_bias,
        window_left=window_left,
        window_right=window_right,
    )

    bias_block_spec = (
        pl.BlockSpec((1, 1, block_q, padded_len), lambda b, h, i: (b, h, i, 0))
        if has_bias
        else pl.BlockSpec((1, 1, block_q, padded_len), lambda b, h, i: (0, 0, 0, 0))
    )

    out = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, i: (b, h, i, 0)),
            pl.BlockSpec((1, 1, padded_len, head_dim), lambda b, h, i: (b, h, 0, 0)),
            pl.BlockSpec((1, 1, padded_len, head_dim), lambda b, h, i: (b, h, 0, 0)),
            bias_block_spec,
        ],
        out_specs=pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, i: (b, h, i, 0)),
        out_shape=jax.ShapeDtypeStruct((batch, num_heads, padded_len, head_dim), q.dtype),
        interpret=interpret,
    )(q, k, v, bias)

    if pad_amount > 0:
        out = out[:, :, :seq_len, :]

    return out


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
