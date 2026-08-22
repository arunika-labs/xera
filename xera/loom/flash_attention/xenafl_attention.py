"""
XeraNaiveFlash Attention.

A from-scratch, pure-`jax.numpy` flash attention: no Pallas, no custom
kernel, no vendor backend. Just block tiling + online (running) softmax,
expressed as plain jnp ops inside `jax.lax.scan`.

It exists as a clean reference for the *algorithm* -- block
tiling + running softmax -- decoupled from any kernel/dispatch concerns.
Being pure jnp also makes it dtype-, device-, and shape-agnostic for
free: whatever `q`/`k`/`v`'s dtype is, jnp ops just use it; whatever
backend XLA picks for the current device, this runs there. There is no
notion of "unsupported dtype" or "unsupported platform" here at all --
that kind of gating belongs to a dispatcher wrapping this, not to the
algorithm itself.

Why a custom VJP despite being pure jnp
----------------------------------------
`jax.grad` can differentiate straight through `jax.lax.scan` + tiling +
online softmax with no help -- every op involved is natively
differentiable. So a custom_vjp is not required for *correctness*.

It is required for the memory target, though. The whole point of flash
attention is O(seq_len) memory instead of O(seq_len^2): the full
(seq_len, seq_len) score/probability matrix is never materialized in one
shot. Autodiff-through-scan does not give you that automatically --
`jax.grad`'s default reverse-mode pass will save (checkpoint) each
scanned iteration's intermediates for reuse on the backward sweep, which
means each key/value block's scores/probabilities end up retained for
the whole backward pass, functionally undoing the tiling's memory
benefit for the exact same reason a naive full-matrix implementation
would.

So this module uses a real `jax.custom_vjp`:
  - Forward: tile over K/V blocks with `jax.lax.scan`, maintain a running
    max, running sum, and running output accumulator (the standard
    online-softmax recurrence), and additionally return the per-query
    log-sum-exp (`lse`) as a residual.
  - Backward: recompute attention probabilities block-by-block from `lse`
    (rather than replaying/storing them), accumulate dQ/dK/dV with
    another `jax.lax.scan`. This is the same two-pass strategy the
    original FlashAffection paper's backward pass uses -- the score
    matrix is never materialized in full on either the forward or the
    backward side, on top of that.

Every array involved is O(seq_len) or O(seq_len * block) in memory (fixed
block size), never O(seq_len^2), by construction.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------

def _resolve_scale(scale, head_dim):
    return scale if scale is not None else 1.0 / (head_dim ** 0.5)


def _num_blocks(seq_len: int, block: int) -> int:
    return -(-seq_len // block)  # ceil division


def _block_bounds(block_idx: int, block: int, seq_len: int):
    """Start offset and validity mask for a fixed-size block that may run
    past `seq_len` on the last iteration (when seq_len isn't a multiple of
    `block`). Returns (start, positions, valid_mask)."""
    start = block_idx * block
    positions = start + jnp.arange(block)
    valid = positions < seq_len
    return start, positions, valid


def _apply_masks(
    scores,          # (block_q, block_k)
    q_positions,      # (block_q,)
    k_positions,      # (block_k,)
    k_valid,          # (block_k,) bool -- False for padding past seq_len
    *,
    causal: bool,
    window_left: int | None,
    window_right: int | None,
):
    """Applies padding / causal / local-window masks to a raw score block,
    all algorithm-level (no dtype/platform assumptions)."""
    mask = jnp.broadcast_to(k_valid[None, :], scores.shape)

    if causal or window_left is not None or window_right is not None:
        rel = q_positions[:, None] - k_positions[None, :]  # (block_q, block_k)
        if causal:
            mask &= rel >= 0
        if window_left is not None:
            mask &= rel <= window_left
        if window_right is not None:
            mask &= -rel <= window_right

    return jnp.where(mask, scores, -jnp.inf)


# ---------------------------------------------------------------------------
# Forward: tiled, online-softmax attention for a single (batch, head).
# Operates on 2D (seq_len, head_dim) q/k/v -- vmapped over batch/heads by
# the public entry point.
# ---------------------------------------------------------------------------

def _forward_single(
    q, k, v, bias,          # (seq_len, head_dim), (seq_len, head_dim), (seq_len, head_dim), (seq_len, seq_len) or None
    *,
    causal: bool,
    scale: float,
    window_left: int | None,
    window_right: int | None,
    block_q: int,
    block_k: int,
):
    seq_len, head_dim = q.shape
    n_k_blocks = _num_blocks(seq_len, block_k)
    n_q_blocks = _num_blocks(seq_len, block_q)

    # Padding is loop-invariant (doesn't depend on the scanned/vmapped block
    # index), so it's done once here rather than inside the scan body --
    # relying on XLA to hoist it out on its own is implementation-defined,
    # not guaranteed.
    q_padded = jnp.pad(q, ((0, block_q), (0, 0)))
    k_padded = jnp.pad(k, ((0, block_k), (0, 0)))
    v_padded = jnp.pad(v, ((0, block_k), (0, 0)))
    bias_padded = jnp.pad(bias, ((0, block_q), (0, block_k))) if bias is not None else None

    def q_block_fn(q_block_idx):
        q_start, q_pos, q_valid = _block_bounds(q_block_idx, block_q, seq_len)
        q_blk = jax.lax.dynamic_slice_in_dim(q_padded, q_start, block_q, axis=0)

        # Running accumulators for the online-softmax recurrence.
        acc0 = jnp.zeros((block_q, head_dim), dtype=jnp.float32)
        m0 = jnp.full((block_q,), -jnp.inf, dtype=jnp.float32)
        l0 = jnp.zeros((block_q,), dtype=jnp.float32)

        def k_block_step(carry, k_block_idx):
            acc, m_prev, l_prev = carry
            k_start, k_pos, k_valid = _block_bounds(k_block_idx, block_k, seq_len)

            k_blk = jax.lax.dynamic_slice_in_dim(k_padded, k_start, block_k, axis=0)
            v_blk = jax.lax.dynamic_slice_in_dim(v_padded, k_start, block_k, axis=0)

            scores = jnp.einsum("qd,kd->qk", q_blk, k_blk).astype(jnp.float32) * scale

            if bias is not None:
                bias_blk = jax.lax.dynamic_slice(
                    bias_padded, (q_start, k_start), (block_q, block_k)
                )
                scores = scores + bias_blk

            scores = _apply_masks(
                scores, q_pos, k_pos, k_valid,
                causal=causal, window_left=window_left, window_right=window_right,
            )

            m_cur = jnp.max(scores, axis=-1)                     # (block_q,)
            m_new = jnp.maximum(m_prev, m_cur)

            # Guard fully-masked rows (m_new == -inf) to avoid exp(-inf-(-inf)) = NaN.
            safe_m_new = jnp.where(jnp.isfinite(m_new), m_new, 0.0)

            p = jnp.exp(scores - safe_m_new[:, None])            # (block_q, block_k)
            p = jnp.where(jnp.isfinite(scores), p, 0.0)

            alpha = jnp.exp(m_prev - safe_m_new)
            alpha = jnp.where(jnp.isfinite(m_prev), alpha, 0.0)

            l_new = alpha * l_prev + jnp.sum(p, axis=-1)
            acc_new = alpha[:, None] * acc + jnp.einsum("qk,kd->qd", p, v_blk)

            return (acc_new, m_new, l_new), None

        (acc, m, l), _ = jax.lax.scan(
            k_block_step, (acc0, m0, l0), jnp.arange(n_k_blocks)
        )

        l_safe = jnp.where(l > 0, l, 1.0)
        out_blk = (acc / l_safe[:, None]).astype(q.dtype)
        lse_blk = (m + jnp.log(l_safe)).astype(jnp.float32)

        # Rows past seq_len are padding; zero them out (never read downstream).
        out_blk = jnp.where(q_valid[:, None], out_blk, 0.0)
        lse_blk = jnp.where(q_valid, lse_blk, 0.0)
        return out_blk, lse_blk

    out_blocks, lse_blocks = jax.vmap(q_block_fn)(jnp.arange(n_q_blocks))
    out = out_blocks.reshape(n_q_blocks * block_q, head_dim)[:seq_len]
    lse = lse_blocks.reshape(n_q_blocks * block_q)[:seq_len]
    return out, lse


# ---------------------------------------------------------------------------
# Backward: recompute probabilities block-by-block from `lse` (never store
# them from the forward pass), tiled the same way as forward.
#
# Two structurally-symmetric passes, exactly like the original FlashAttention
# backward: pass 1 (vmap over k-block, scan over q-block) accumulates dK/dV;
# pass 2 (vmap over q-block, scan over k-block) accumulates dQ. Both passes
# recompute `p`/`dp`/`ds` from the residuals (q, k, v, bias, lse, delta)
# rather than reusing tiles from the other pass -- reusing would mean
# materializing every (q_block, k_block) `ds` tile at once, which is exactly
# the full (seq_len, seq_len) probability-gradient matrix (just chunked),
# silently regressing memory to O(seq_len^2). Recomputing costs one extra
# score matmul (the same trade-off the original paper makes) but keeps both
# passes O(seq_len) / O(seq_len * block), matching the module's memory claim.
#
# The one legitimate exception is `dbias`: its own memory footprint is
# already O(seq_len^2) by definition (it's a full attention-shaped array),
# so pass 1 additionally emits its `ds` tiles as scan outputs when a bias is
# present, and they're reassembled into the dense (seq_len, seq_len)
# gradient via a single transpose+reshape (no per-tile Python loop).
# ---------------------------------------------------------------------------

def _backward_single(
    q, k, v, bias, out, lse, d_out,
    *,
    causal: bool,
    scale: float,
    window_left: int | None,
    window_right: int | None,
    block_q: int,
    block_k: int,
):
    seq_len, head_dim = q.shape
    n_k_blocks = _num_blocks(seq_len, block_k)
    n_q_blocks = _num_blocks(seq_len, block_q)
    has_bias = bias is not None

    # delta_i = sum_d(out_i,d * d_out_i,d) -- the standard flash-attention
    # backward trick, avoids ever materializing the full probability
    # gradient in one shot.
    delta = jnp.sum(out.astype(jnp.float32) * d_out.astype(jnp.float32), axis=-1)  # (seq_len,)

    # Padding is loop-invariant, so it's done once here rather than inside
    # scan bodies (see same note in `_forward_single`).
    q_padded = jnp.pad(q, ((0, block_q), (0, 0)))
    k_padded = jnp.pad(k, ((0, block_k), (0, 0)))
    v_padded = jnp.pad(v, ((0, block_k), (0, 0)))
    d_out_padded = jnp.pad(d_out, ((0, block_q), (0, 0)))
    lse_padded = jnp.pad(lse, ((0, block_q),))
    delta_padded = jnp.pad(delta, ((0, block_q),))
    bias_padded = jnp.pad(bias, ((0, block_q), (0, block_k))) if has_bias else None

    def _recompute_p(q_blk, k_blk, bias_blk, q_pos, k_pos, k_valid, q_valid, lse_blk):
        """Recomputes the (block_q, block_k) softmax-probability tile from
        residuals -- shared by both passes so they stay numerically
        identical."""
        scores = jnp.einsum("qd,kd->qk", q_blk, k_blk).astype(jnp.float32) * scale
        if has_bias:
            scores = scores + bias_blk
        scores = _apply_masks(
            scores, q_pos, k_pos, k_valid,
            causal=causal, window_left=window_left, window_right=window_right,
        )
        p = jnp.exp(scores - lse_blk[:, None])
        p = jnp.where(jnp.isfinite(scores), p, 0.0)
        p = jnp.where(q_valid[:, None], p, 0.0)
        return p

    # --- Pass 1: dK, dV (+ ds tiles for dbias, if a bias was given) --------
    def kv_block_fn(k_block_idx):
        k_start, k_pos, k_valid = _block_bounds(k_block_idx, block_k, seq_len)
        k_blk = jax.lax.dynamic_slice_in_dim(k_padded, k_start, block_k, axis=0)
        v_blk = jax.lax.dynamic_slice_in_dim(v_padded, k_start, block_k, axis=0)

        dk0 = jnp.zeros((block_k, head_dim), dtype=jnp.float32)
        dv0 = jnp.zeros((block_k, head_dim), dtype=jnp.float32)

        def q_block_step(carry, q_block_idx):
            dk, dv = carry
            q_start, q_pos, q_valid = _block_bounds(q_block_idx, block_q, seq_len)

            q_blk = jax.lax.dynamic_slice_in_dim(q_padded, q_start, block_q, axis=0)
            d_out_blk = jax.lax.dynamic_slice_in_dim(d_out_padded, q_start, block_q, axis=0)
            lse_blk = jax.lax.dynamic_slice_in_dim(lse_padded, q_start, block_q, axis=0)
            delta_blk = jax.lax.dynamic_slice_in_dim(delta_padded, q_start, block_q, axis=0)

            bias_blk = None
            if has_bias:
                bias_blk = jax.lax.dynamic_slice(
                    bias_padded, (q_start, k_start), (block_q, block_k)
                )

            p = _recompute_p(q_blk, k_blk, bias_blk, q_pos, k_pos, k_valid, q_valid, lse_blk)

            d_out_blk_f32 = d_out_blk.astype(jnp.float32)
            dv_contrib = jnp.einsum("qk,qd->kd", p, d_out_blk_f32)

            dp = jnp.einsum("qd,kd->qk", d_out_blk_f32, v_blk.astype(jnp.float32))
            # d(score)/d(bias) = 1 (bias enters additively, post-scale), so
            # ds_raw is the gradient w.r.t. bias directly. d(score)/d(q@k)
            # picks up the extra `scale` factor from `scores = q@k * scale
            # + bias`, so dk (and dq, in pass 2) needs the scaled version.
            ds_raw = p * (dp - delta_blk[:, None])              # (block_q, block_k)
            ds_scaled = ds_raw * scale

            dk_contrib = jnp.einsum("qk,qd->kd", ds_scaled, q_blk.astype(jnp.float32))

            new_carry = (dk + dk_contrib, dv + dv_contrib)
            # Only emit the full ds tile when it's actually needed for
            # dbias -- otherwise pass 1 stays O(seq_len) in its outputs too.
            ys = ds_raw if has_bias else None
            return new_carry, ys

        (dk, dv), ds_all = jax.lax.scan(
            q_block_step, (dk0, dv0), jnp.arange(n_q_blocks)
        )

        dk = jnp.where(k_valid[:, None], dk, 0.0)
        dv = jnp.where(k_valid[:, None], dv, 0.0)
        return dk, dv, ds_all

    dk_blocks, dv_blocks, ds_all_blocks = jax.vmap(kv_block_fn)(jnp.arange(n_k_blocks))

    dk = dk_blocks.reshape(n_k_blocks * block_k, head_dim)[:seq_len].astype(k.dtype)
    dv = dv_blocks.reshape(n_k_blocks * block_k, head_dim)[:seq_len].astype(v.dtype)

    # --- Pass 2: dQ, structurally symmetric to pass 1 -- recomputes `p`/
    # `dp`/`ds` per (q_block, k_block) tile instead of reusing pass 1's
    # `ds_all_blocks`, so this pass never holds more than one tile at a
    # time (O(seq_len) memory, matching the module's docstring). -----------
    def q_block_dq(q_block_idx):
        q_start, q_pos, q_valid = _block_bounds(q_block_idx, block_q, seq_len)
        q_blk = jax.lax.dynamic_slice_in_dim(q_padded, q_start, block_q, axis=0)
        d_out_blk = jax.lax.dynamic_slice_in_dim(d_out_padded, q_start, block_q, axis=0)
        lse_blk = jax.lax.dynamic_slice_in_dim(lse_padded, q_start, block_q, axis=0)
        delta_blk = jax.lax.dynamic_slice_in_dim(delta_padded, q_start, block_q, axis=0)
        d_out_blk_f32 = d_out_blk.astype(jnp.float32)

        dq0 = jnp.zeros((block_q, head_dim), dtype=jnp.float32)

        def k_block_step(dq, k_block_idx):
            k_start, k_pos, k_valid = _block_bounds(k_block_idx, block_k, seq_len)
            k_blk = jax.lax.dynamic_slice_in_dim(k_padded, k_start, block_k, axis=0)
            v_blk = jax.lax.dynamic_slice_in_dim(v_padded, k_start, block_k, axis=0)

            bias_blk = None
            if has_bias:
                bias_blk = jax.lax.dynamic_slice(
                    bias_padded, (q_start, k_start), (block_q, block_k)
                )

            p = _recompute_p(q_blk, k_blk, bias_blk, q_pos, k_pos, k_valid, q_valid, lse_blk)

            dp = jnp.einsum("qd,kd->qk", d_out_blk_f32, v_blk.astype(jnp.float32))
            ds_raw = p * (dp - delta_blk[:, None])
            ds_scaled = ds_raw * scale

            dq_contrib = jnp.einsum("qk,kd->qd", ds_scaled, k_blk.astype(jnp.float32))
            return dq + dq_contrib, None

        dq, _ = jax.lax.scan(k_block_step, dq0, jnp.arange(n_k_blocks))
        dq = jnp.where(q_valid[:, None], dq, 0.0)
        return dq

    dq_blocks = jax.vmap(q_block_dq)(jnp.arange(n_q_blocks))
    dq = dq_blocks.reshape(n_q_blocks * block_q, head_dim)[:seq_len].astype(q.dtype)

    if has_bias:
        # dbias: for each (q, k) pair, ds is exactly d(bias). Reassemble
        # from the per-(k_block, q_block) ds tiles into a dense
        # (seq_len, seq_len) gradient with a single transpose+reshape --
        # the blocks are non-overlapping and cover the padded grid exactly,
        # so this needs no per-tile scatter/loop. bias's own memory
        # footprint is already O(seq_len^2) by definition (it's a full
        # attention-shaped array), so this doesn't regress the O(seq_len)
        # target for q/k/v.
        # ds_all_blocks: (n_k_blocks, n_q_blocks, block_q, block_k)
        dbias_full = ds_all_blocks.transpose(1, 2, 0, 3).reshape(
            n_q_blocks * block_q, n_k_blocks * block_k
        )
        dbias = dbias_full[:seq_len, :seq_len].astype(bias.dtype)
    else:
        dbias = None

    return dq, dk, dv, dbias


# ---------------------------------------------------------------------------
# Public, differentiable entry point. vmapped over (batch, num_heads).
# ---------------------------------------------------------------------------

@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9))
def xenafl_attention(
    q, k, v, bias,
    causal, scale, window_left, window_right, block_q, block_k,
):
    """
    XeraNaiveFlash attention: pure-jnp tiled attention with online softmax.

    O(seq_len) memory in both the forward and backward pass (fixed block
    size), by construction -- the full (seq_len, seq_len) score/probability
    matrix is never materialized in one shot on either side. No kernel, no
    Pallas, no vendor backend: just `jnp` ops inside `jax.lax.scan`, so
    this is dtype-, device-, and platform-agnostic -- it runs anywhere JAX
    does, on whatever dtype `q`/`k`/`v` already are.

    Args:
        q, k, v: (batch, num_heads, seq_len, head_dim) arrays.
        bias: Optional additive attention bias, broadcastable to
            (batch, num_heads, seq_len, seq_len), or None.
        causal: If True, apply a causal mask (query i attends to key j
            only if j <= i).
        scale: Softmax scale. If None, defaults to 1/sqrt(head_dim).
        window_left, window_right: Local attention window bounds (query i
            may attend to key j only if `i - j <= window_left` and
            `j - i <= window_right`), or None for unbounded on that side.
            Combine with `causal` if both given.
        block_q, block_k: Tile sizes along the query/key sequence
            dimension. Any positive int; seq_len need not be a multiple
            of either.

    Returns:
        Output array of shape (batch, num_heads, seq_len, head_dim), same
        dtype as `q`.
    """
    out, _lse = _xenafl_forward_impl(
        q, k, v, bias,
        causal=causal, scale=scale, window_left=window_left, window_right=window_right,
        block_q=block_q, block_k=block_k,
    )
    return out


def _xenafl_forward_impl(
    q, k, v, bias, *, causal, scale, window_left, window_right, block_q, block_k,
):
    _batch, _heads, _seq_len, head_dim = q.shape
    resolved_scale = _resolve_scale(scale, head_dim)

    fwd = functools.partial(
        _forward_single,
        causal=causal, scale=resolved_scale,
        window_left=window_left, window_right=window_right,
        block_q=block_q, block_k=block_k,
    )

    if bias is None:
        batched = jax.vmap(jax.vmap(lambda q_, k_, v_: fwd(q_, k_, v_, None)))
        out, lse = batched(q, k, v)
    else:
        bias_b = jnp.broadcast_to(bias, (q.shape[0], q.shape[1], q.shape[2], q.shape[2]))
        batched = jax.vmap(jax.vmap(fwd))
        out, lse = batched(q, k, v, bias_b)

    return out, lse


def _xenafl_fwd(q, k, v, bias, causal, scale, window_left, window_right, block_q, block_k):
    out, lse = _xenafl_forward_impl(
        q, k, v, bias,
        causal=causal, scale=scale, window_left=window_left, window_right=window_right,
        block_q=block_q, block_k=block_k,
    )
    residuals = (q, k, v, bias, out, lse)
    return out, residuals


def _xenafl_bwd(causal, scale, window_left, window_right, block_q, block_k, residuals, d_out):
    q, k, v, bias, out, lse = residuals
    _batch, _heads, _seq_len, head_dim = q.shape
    resolved_scale = _resolve_scale(scale, head_dim)

    bwd = functools.partial(
        _backward_single,
        causal=causal, scale=resolved_scale,
        window_left=window_left, window_right=window_right,
        block_q=block_q, block_k=block_k,
    )

    if bias is None:
        batched = jax.vmap(jax.vmap(
            lambda q_, k_, v_, out_, lse_, do_: bwd(q_, k_, v_, None, out_, lse_, do_)
        ))
        dq, dk, dv, _ = batched(q, k, v, out, lse, d_out)
        dbias = None
    else:
        bias_b = jnp.broadcast_to(bias, (q.shape[0], q.shape[1], q.shape[2], q.shape[2]))
        batched = jax.vmap(jax.vmap(bwd))
        dq, dk, dv, dbias_full = batched(q, k, v, bias_b, out, lse, d_out)
        # Undo the broadcast: sum gradient back down to bias's original shape.
        dbias = _unbroadcast(dbias_full, bias.shape)

    return dq, dk, dv, dbias


def _unbroadcast(grad, target_shape):
    """Sums `grad` back down to `target_shape` along any axis that was
    broadcast (size-1 in target_shape, or a leading axis target_shape
    doesn't have at all)."""
    ndim_diff = grad.ndim - len(target_shape)
    axes_to_sum = tuple(range(ndim_diff))
    padded_target = (1,) * ndim_diff + tuple(target_shape)
    axes_to_sum += tuple(
        i for i, (g_dim, t_dim) in enumerate(zip(grad.shape, padded_target))
        if t_dim == 1 and g_dim != 1
    )
    if axes_to_sum:
        grad = jnp.sum(grad, axis=axes_to_sum, keepdims=True)
    return grad.reshape(target_shape)


xenafl_attention.defvjp(_xenafl_fwd, _xenafl_bwd)


__all__ = ["xenafl_attention"]
