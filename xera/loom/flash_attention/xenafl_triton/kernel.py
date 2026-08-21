"""
XeraNaiveFlash, Triton edition -- forward and backward kernels.

Same algorithm as `xenafl_attention.py` (block tiling + online softmax,
O(seq_len) memory, two-pass backward that recomputes probabilities from
the saved log-sum-exp rather than storing them) -- written in
`triton.language` instead of `jnp`, so it can run as a real fused GPU
kernel rather than a sequence of separate XLA ops. See that module's
docstring for the algorithmic rationale (why a custom backward is
needed at all, why recompute-from-lse rather than store-and-replay).

This file was written from scratch for `xera`. The kernel *shape* --
one program-instance-per-(query-block, batch*head) grid for forward,
and a program-instance-per-(key/value-block, batch*head) grid for
backward that loops over the query blocks each KV block interacts with,
accumulating dK/dV locally and dQ via `tl.atomic_add` into a global
buffer -- is the standard FA1-style Triton kernel shape (this is how
the algorithm maps onto Triton's grid model; the atomic-add-to-dQ
pattern in particular is the well-established way to avoid either
storing the full score matrix or requiring a third kernel pass). Every
kernel here is xera's own code: named, organized, and feature-gated
(causal / local window / segment_ids as constexpr flags) to mirror
`xenafl_attention.py`'s structure and feature set one-to-one.

Supported (mirrors xenafl_attention.py, minus `bias` -- see TODO below):
    - causal masking
    - local window (window_left, window_right)
    - segment_ids (packing: query i and key j attend to each other only
      if segment_ids[i] == segment_ids[j])

Not yet supported here (present in xenafl_attention.py):
    - additive `bias` -- needs an extra pointer + stride plumbed through
      both kernels; deferred until the no-bias path is validated on
      real GPU hardware, since bias adds a second memory-bound load per
      block and is easiest to get right once the core recurrence is
      confirmed correct.

** UNTESTED: there is no NVIDIA GPU available in the environment this
was written in, so these kernels have NOT been executed even once. **
They were written carefully against the algorithm already validated in
`xenafl_attention.py` (same recurrence, same two-pass backward) and
against the standard FA1 Triton kernel shape (including the
atomic-add-to-dQ backward pattern, cross-checked against a public
reference implementation for kernel *shape* -- grid layout, pointer
arithmetic, the accumulator recurrence -- while writing this; no code
was copied, and the feature set/masking here is xera's own, following
`xenafl_attention.py` rather than that reference). "Carefully derived"
is not a substitute for running it. Before trusting this for anything:
run a parity test against `xenafl_attention`'s forward output and
`jax.grad`-based reference gradients (mirroring
`test_xenafl_attention.py`'s existing tests) on actual CUDA hardware
first, ideally sm_70/sm_75 since that's this backend's primary target.
Pay special attention to the dQ atomic-add path (float32 accumulation
buffer, cast to the input dtype only after all KV blocks have
contributed) and to num_warps/num_stages tuning, neither of which can
be validated without a real GPU.
"""

from __future__ import annotations

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel.
#
# Grid: (num_q_blocks, batch * num_heads). Each program instance handles
# one BLOCK_Q-sized slice of queries for one (batch, head) pair, looping
# over all key/value blocks that slice attends to.
# ---------------------------------------------------------------------------

@triton.jit
def _xenafl_triton_fwd_kernel(
    q_ptr, k_ptr, v_ptr,
    out_ptr, lse_ptr,
    segment_ids_ptr,
    seq_len,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    stride_sb, stride_sm,
    num_heads,
    CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    HAS_SEGMENT_IDS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    batch_idx = bh_idx // num_heads
    head_idx = bh_idx % num_heads

    q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offsets = tl.arange(0, BLOCK_D)
    q_valid = q_offsets < seq_len

    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    q_ptrs = q_base + q_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)

    if HAS_SEGMENT_IDS:
        seg_base = segment_ids_ptr + batch_idx * stride_sb
        q_segment = tl.load(seg_base + q_offsets * stride_sm, mask=q_valid, other=-1)

    # Online-softmax running state -- same recurrence as
    # `xenafl_attention._forward_single`'s `k_block_step`.
    m_i = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    # Causal early-exit: skip key blocks strictly past this query
    # block's end. Purely an optimization (correctness comes from the
    # per-element mask below regardless); mirrors the standard Triton
    # FA kernel shape.
    if CAUSAL:
        hi = tl.minimum((q_block_idx + 1) * BLOCK_Q, seq_len)
    else:
        hi = seq_len

    k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + head_idx * stride_vh

    for k_start in range(0, hi, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_valid = k_offsets < seq_len

        k_ptrs = k_base + k_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd
        v_ptrs = v_base + k_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd
        k_block = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0)
        v_block = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)

        scores = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * sm_scale

        mask = k_valid[None, :]
        if CAUSAL:
            mask = mask & (q_offsets[:, None] >= k_offsets[None, :])
        if HAS_WINDOW:
            rel = q_offsets[:, None] - k_offsets[None, :]
            mask = mask & (rel <= WINDOW_LEFT) & (-rel <= WINDOW_RIGHT)
        if HAS_SEGMENT_IDS:
            k_segment = tl.load(seg_base + k_offsets * stride_sm, mask=k_valid, other=-2)
            mask = mask & (q_segment[:, None] == k_segment[None, :])

        scores = tl.where(mask, scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        safe_m_new = tl.where(m_new == -float("inf"), 0.0, m_new)

        p = tl.exp(scores - safe_m_new[:, None])
        p = tl.where(scores == -float("inf"), 0.0, p)

        alpha = tl.exp(m_i - safe_m_new)
        alpha = tl.where(m_i == -float("inf"), 0.0, alpha)

        l_new = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)

        m_i = m_new
        l_i = l_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out_block = acc / l_safe[:, None]
    lse = m_i + tl.log(l_safe)

    out_base = out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    out_ptrs = out_base + q_offsets[:, None] * stride_om + d_offsets[None, :] * stride_od
    tl.store(out_ptrs, out_block.to(q_block.dtype), mask=q_valid[:, None])

    lse_base = lse_ptr + batch_idx * stride_lb + head_idx * stride_lh
    tl.store(lse_base + q_offsets * stride_lm, lse, mask=q_valid)


# ---------------------------------------------------------------------------
# Backward preprocess: delta_i = sum_d(out_i,d * dout_i,d).
#
# The standard flash-attention backward trick -- lets dP be computed
# block-by-block in the main backward kernel without ever materializing
# the full (seq_len, seq_len) probability gradient. Mirrors the `delta`
# computation at the top of `xenafl_attention._backward_single`.
# ---------------------------------------------------------------------------

@triton.jit
def _xenafl_triton_bwd_preprocess(
    out_ptr, dout_ptr, delta_ptr,
    seq_len,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_db, stride_dh, stride_dm,
    num_heads,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    batch_idx = bh_idx // num_heads
    head_idx = bh_idx % num_heads

    q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offsets = tl.arange(0, BLOCK_D)
    q_valid = q_offsets < seq_len

    out_base = out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    dout_base = dout_ptr + batch_idx * stride_ob + head_idx * stride_oh
    ptrs = q_offsets[:, None] * stride_om + d_offsets[None, :] * stride_od

    o = tl.load(out_base + ptrs, mask=q_valid[:, None], other=0.0).to(tl.float32)
    do = tl.load(dout_base + ptrs, mask=q_valid[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)

    delta_base = delta_ptr + batch_idx * stride_db + head_idx * stride_dh
    tl.store(delta_base + q_offsets * stride_dm, delta, mask=q_valid)


# ---------------------------------------------------------------------------
# Main backward kernel.
#
# Grid: (num_k_blocks, batch * num_heads). Each program instance owns
# one BLOCK_K-sized slice of keys/values for one (batch, head) pair,
# and loops over every query block that slice interacts with --
# mirroring `xenafl_attention._backward_single`'s `kv_block_fn` (outer)
# / `q_block_step` (inner) nesting.
#
# dK/dV for this KV block are accumulated locally across the Q loop and
# stored once at the end (each program instance owns its own KV rows
# exclusively, so no atomics needed there). dQ, however, is written to
# by every KV block that any given Q block attends to -- i.e. multiple
# program instances contend for the same dQ rows -- so dQ is
# accumulated into a float32 global buffer via `tl.atomic_add`. The
# caller (`dispatch.py`) must zero-initialize that dQ buffer before
# launching this kernel, and cast it back to the input dtype afterward.
# ---------------------------------------------------------------------------

@triton.jit
def _xenafl_triton_bwd_kernel(
    q_ptr, k_ptr, v_ptr, dout_ptr,
    lse_ptr, delta_ptr, segment_ids_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    seq_len,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_lb, stride_lh, stride_lm,
    stride_sb, stride_sm,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,  # dq_ptr's own strides -- float32 buffer, may differ from q's
    num_heads,
    CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    HAS_SEGMENT_IDS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    k_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    batch_idx = bh_idx // num_heads
    head_idx = bh_idx % num_heads

    d_offsets = tl.arange(0, BLOCK_D)
    k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    k_valid = k_offsets < seq_len

    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + head_idx * stride_vh
    dout_base = dout_ptr + batch_idx * stride_qb + head_idx * stride_qh
    lse_base = lse_ptr + batch_idx * stride_lb + head_idx * stride_lh
    delta_base = delta_ptr + batch_idx * stride_lb + head_idx * stride_lh
    dq_base = dq_ptr + batch_idx * stride_dqb + head_idx * stride_dqh
    dk_base = dk_ptr + batch_idx * stride_kb + head_idx * stride_kh
    dv_base = dv_ptr + batch_idx * stride_vb + head_idx * stride_vh
    if HAS_SEGMENT_IDS:
        seg_base = segment_ids_ptr + batch_idx * stride_sb
        k_segment = tl.load(seg_base + k_offsets * stride_sm, mask=k_valid, other=-2)

    k_ptrs = k_base + k_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd
    v_ptrs = v_base + k_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd
    k_block = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0)
    v_block = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)

    dk_acc = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)

    # Causal: this KV block only receives gradient from Q blocks at or
    # after it (query i attends to key j only if i >= j).
    if CAUSAL:
        q_lo = (k_block_idx * BLOCK_K // BLOCK_Q) * BLOCK_Q
    else:
        q_lo = 0

    for q_start in range(q_lo, seq_len, BLOCK_Q):
        q_offsets = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_offsets < seq_len

        q_ptrs = q_base + q_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
        dout_ptrs = dout_base + q_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
        q_block = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)
        do_block = tl.load(dout_ptrs, mask=q_valid[:, None], other=0.0).to(tl.float32)

        lse_block = tl.load(lse_base + q_offsets * stride_lm, mask=q_valid, other=0.0)
        delta_block = tl.load(delta_base + q_offsets * stride_lm, mask=q_valid, other=0.0)

        scores = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * sm_scale

        mask = k_valid[None, :] & q_valid[:, None]
        if CAUSAL:
            mask = mask & (q_offsets[:, None] >= k_offsets[None, :])
        if HAS_WINDOW:
            rel = q_offsets[:, None] - k_offsets[None, :]
            mask = mask & (rel <= WINDOW_LEFT) & (-rel <= WINDOW_RIGHT)
        if HAS_SEGMENT_IDS:
            q_segment = tl.load(seg_base + q_offsets * stride_sm, mask=q_valid, other=-1)
            mask = mask & (q_segment[:, None] == k_segment[None, :])

        # Recompute P from the saved LSE rather than storing it from the
        # forward pass -- same trick as `xenafl_attention._backward_single`.
        p = tl.exp(scores - lse_block[:, None])
        p = tl.where(mask, p, 0.0)

        dv_acc += tl.dot(tl.trans(p.to(do_block.dtype)), do_block)

        dp = tl.dot(do_block, tl.trans(v_block).to(do_block.dtype)).to(tl.float32)
        ds = p * (dp - delta_block[:, None])   # dScore, pre-scale (== dBias, if bias existed)
        ds_scaled = ds * sm_scale                # scaled, for dQ/dK (chain rule through `* sm_scale`)

        dk_acc += tl.dot(tl.trans(ds_scaled.to(q_block.dtype)), q_block)

        dq_contrib = tl.dot(ds_scaled.to(k_block.dtype), k_block).to(tl.float32)
        dq_ptrs = dq_base + q_offsets[:, None] * stride_dqm + d_offsets[None, :] * stride_dqd
        tl.atomic_add(dq_ptrs, dq_contrib, mask=q_valid[:, None])

    dk_ptrs = dk_base + k_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd
    dv_ptrs = dv_base + k_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd
    tl.store(dk_ptrs, dk_acc.to(k_block.dtype), mask=k_valid[:, None])
    tl.store(dv_ptrs, dv_acc.to(v_block.dtype), mask=k_valid[:, None])
