"""Tests for xera.loom.xenafl_attention: pure-jnp tiled attention with
online softmax (forward correctness and custom_vjp gradient correctness),
checked against a plain non-tiled reference implementation."""

import jax
import jax.numpy as jnp
import pytest
from xera.loom.xenafl_attention import xenafl_attention


def reference_attention(q, k, v, *, causal=False, scale=None, bias=None, window_left=None, window_right=None):
    head_dim = q.shape[-1]
    seq_len = q.shape[2]
    scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)
    scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
    if bias is not None:
        scores = scores + bias
    if causal:
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(mask, scores, -jnp.inf)
    if window_left is not None or window_right is not None:
        q_pos = jnp.arange(seq_len)[:, None]
        k_pos = jnp.arange(seq_len)[None, :]
        rel = q_pos - k_pos
        window_mask = jnp.ones((seq_len, seq_len), dtype=bool)
        if window_left is not None:
            window_mask &= rel <= window_left
        if window_right is not None:
            window_mask &= -rel <= window_right
        scores = jnp.where(window_mask, scores, -jnp.inf)
    attn = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhts,bhsd->bhtd", attn, v)


def _make_qkv(key, batch, heads, seq_len, head_dim, dtype=jnp.float32):
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (batch, heads, seq_len, head_dim)).astype(dtype)
    k = jax.random.normal(kk, (batch, heads, seq_len, head_dim)).astype(dtype)
    v = jax.random.normal(kv, (batch, heads, seq_len, head_dim)).astype(dtype)
    return q, k, v


def _call(q, k, v, bias=None, causal=False, scale=None, window_left=None, window_right=None, block_q=16, block_k=16):
    return xenafl_attention(q, k, v, bias, causal, scale, window_left, window_right, block_q, block_k)


# ---------------------------------------------------------------------------
# Forward correctness.
# ---------------------------------------------------------------------------

def test_matches_reference_plain():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 4, 37, 16)
    out = _call(q, k, v, block_q=16, block_k=16)
    ref = reference_attention(q, k, v)
    assert jnp.allclose(out, ref, atol=1e-4)


def test_matches_reference_causal():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 4, 37, 16)
    out = _call(q, k, v, causal=True, block_q=16, block_k=16)
    ref = reference_attention(q, k, v, causal=True)
    assert jnp.allclose(out, ref, atol=1e-4)


def test_matches_reference_with_bias():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 3, 37, 16)
    bias = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 37, 37)) * 0.1
    out = _call(q, k, v, bias=bias, block_q=16, block_k=16)
    ref = reference_attention(q, k, v, bias=bias)
    assert jnp.allclose(out, ref, atol=1e-4)


@pytest.mark.parametrize("window_left,window_right", [(5, 5), (5, None), (None, 5), (0, 0)])
def test_matches_reference_local_window(window_left, window_right):
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 3, 37, 16)
    out = _call(q, k, v, window_left=window_left, window_right=window_right, block_q=16, block_k=16)
    ref = reference_attention(q, k, v, window_left=window_left, window_right=window_right)
    assert jnp.allclose(out, ref, atol=1e-4)


@pytest.mark.parametrize(
    "batch,heads,seq_len,head_dim,block_q,block_k",
    [
        (2, 4, 37, 16, 16, 16),   # seq_len not divisible by either block size
        (1, 1, 8, 4, 8, 8),       # single block
        (2, 4, 64, 16, 16, 16),   # evenly divisible
        (1, 2, 100, 8, 32, 17),   # mismatched, irregular block sizes
        (1, 1, 1, 4, 8, 8),       # seq_len smaller than block size
        (1, 1, 5, 4, 8, 8),       # seq_len smaller than block size
    ],
)
def test_matches_reference_various_shapes(batch, heads, seq_len, head_dim, block_q, block_k):
    q, k, v = _make_qkv(jax.random.PRNGKey(0), batch, heads, seq_len, head_dim)
    for causal in (False, True):
        out = _call(q, k, v, causal=causal, block_q=block_q, block_k=block_k)
        ref = reference_attention(q, k, v, causal=causal)
        assert jnp.allclose(out, ref, atol=1e-4), f"mismatch: causal={causal}"


def test_output_shape_and_dtype_match_input():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 3, 2, 29, 8, dtype=jnp.bfloat16)
    out = _call(q, k, v, block_q=16, block_k=16)
    assert out.shape == q.shape
    assert out.dtype == q.dtype


def test_no_nans_with_narrow_window():
    # A k-block entirely masked out (fully outside the local window) must
    # not produce NaNs via the online-softmax running-max update.
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 1, 1, 37, 4)
    out = _call(q, k, v, window_left=5, window_right=5, block_q=16, block_k=16)
    assert not jnp.any(jnp.isnan(out))


# ---------------------------------------------------------------------------
# Gradient correctness (custom_vjp vs autodiff over the reference impl).
# ---------------------------------------------------------------------------

def test_gradients_match_reference_causal():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 3, 29, 8)

    def loss_xenafl(q, k, v):
        return jnp.sum(_call(q, k, v, causal=True, block_q=16, block_k=16) ** 2)

    def loss_ref(q, k, v):
        return jnp.sum(reference_attention(q, k, v, causal=True) ** 2)

    g_x = jax.grad(loss_xenafl, argnums=(0, 1, 2))(q, k, v)
    g_r = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
    for gx, gr in zip(g_x, g_r):
        assert jnp.allclose(gx, gr, atol=1e-3)


def test_gradients_match_reference_with_bias():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 3, 29, 8)
    bias = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 29, 29)) * 0.1

    def loss_xenafl(q, k, v, bias):
        return jnp.sum(_call(q, k, v, bias=bias, block_q=16, block_k=16) ** 2)

    def loss_ref(q, k, v, bias):
        return jnp.sum(reference_attention(q, k, v, bias=bias) ** 2)

    g_x = jax.grad(loss_xenafl, argnums=(0, 1, 2, 3))(q, k, v, bias)
    g_r = jax.grad(loss_ref, argnums=(0, 1, 2, 3))(q, k, v, bias)
    for name, gx, gr in zip(("dq", "dk", "dv", "dbias"), g_x, g_r):
        assert jnp.allclose(gx, gr, atol=1e-3), f"{name} mismatch"


def test_gradients_match_reference_causal_and_bias():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 3, 29, 8)
    bias = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 29, 29)) * 0.1

    def loss_xenafl(q, k, v, bias):
        return jnp.sum(_call(q, k, v, bias=bias, causal=True, block_q=16, block_k=16) ** 2)

    def loss_ref(q, k, v, bias):
        return jnp.sum(reference_attention(q, k, v, bias=bias, causal=True) ** 2)

    g_x = jax.grad(loss_xenafl, argnums=(0, 1, 2, 3))(q, k, v, bias)
    g_r = jax.grad(loss_ref, argnums=(0, 1, 2, 3))(q, k, v, bias)
    for name, gx, gr in zip(("dq", "dk", "dv", "dbias"), g_x, g_r):
        assert jnp.allclose(gx, gr, atol=1e-3), f"{name} mismatch"


def test_gradients_match_reference_local_window():
    q, k, v = _make_qkv(jax.random.PRNGKey(2), 1, 2, 40, 8)

    def loss_xenafl(q, k, v):
        return jnp.sum(_call(q, k, v, window_left=5, window_right=5, block_q=16, block_k=16) ** 2)

    def loss_ref(q, k, v):
        return jnp.sum(reference_attention(q, k, v, window_left=5, window_right=5) ** 2)

    g_x = jax.grad(loss_xenafl, argnums=(0, 1, 2))(q, k, v)
    g_r = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
    for gx, gr in zip(g_x, g_r):
        assert jnp.allclose(gx, gr, atol=1e-3)


# ---------------------------------------------------------------------------
# jax.jit compatibility.
# ---------------------------------------------------------------------------

def test_jit_forward_matches_reference():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 2, 20, 8)
    jitted = jax.jit(lambda q, k, v: _call(q, k, v, causal=True, block_q=16, block_k=16))
    out = jitted(q, k, v)
    ref = reference_attention(q, k, v, causal=True)
    assert jnp.allclose(out, ref, atol=1e-4)


def test_jit_grad_runs():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 2, 20, 8)
    jit_grad = jax.jit(jax.grad(
        lambda q, k, v: jnp.sum(_call(q, k, v, causal=True, block_q=16, block_k=16) ** 2),
        argnums=(0, 1, 2),
    ))
    g = jit_grad(q, k, v)
    assert all(gi.shape == qi.shape for gi, qi in zip(g, (q, k, v)))
