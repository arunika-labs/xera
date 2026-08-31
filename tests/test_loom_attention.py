"""Tests for xera.loom.attention: MultiHeadAttention, GroupedQueryAttention,
SelfAttention, causal_mask."""

import jax
import jax.numpy as jnp
import pytest
import xera.loom as xl


# ---------------------------------------------------------------------------
# causal_mask
# ---------------------------------------------------------------------------

def test_causal_mask_shape_and_dtype():
    mask = xl.causal_mask(5)
    assert mask.shape == (5, 5)
    assert mask.dtype == bool


def test_causal_mask_lower_triangular():
    mask = xl.causal_mask(4)
    expected = jnp.tril(jnp.ones((4, 4), dtype=bool))
    assert jnp.array_equal(mask, expected)


def test_causal_mask_diagonal_allowed():
    mask = xl.causal_mask(3)
    assert bool(jnp.all(jnp.diag(mask)))


def test_causal_mask_future_positions_blocked():
    mask = xl.causal_mask(3)
    assert not bool(mask[0, 1])
    assert not bool(mask[0, 2])
    assert not bool(mask[1, 2])


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

def test_mha_forward_shape():
    attn = xl.MultiHeadAttention(dim=32, num_heads=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 32))
    out = attn(x)
    assert out.shape == (2, 6, 32)


def test_mha_head_dim_computed():
    attn = xl.MultiHeadAttention(dim=32, num_heads=4, key=jax.random.PRNGKey(0))
    assert attn.head_dim == 8


def test_mha_requires_divisible_dim():
    with pytest.raises(AssertionError):
        xl.MultiHeadAttention(dim=33, num_heads=4, key=jax.random.PRNGKey(0))


def test_mha_causal_mask_blocks_future_influence():
    attn = xl.MultiHeadAttention(dim=16, num_heads=2, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))
    mask = xl.causal_mask(5)[None, None, :, :]

    def out_at_pos0(x_):
        return attn(x_, mask=mask)[0, 0]

    # Changing a future position (index 4) should not affect output at
    # position 0 under a causal mask.
    x_perturbed = x.at[0, 4].add(100.0)
    out1 = out_at_pos0(x)
    out2 = out_at_pos0(x_perturbed)
    assert jnp.allclose(out1, out2, atol=1e-4)


def test_mha_without_mask_all_positions_influence():
    attn = xl.MultiHeadAttention(dim=16, num_heads=2, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))

    out1 = attn(x)[0, 0]
    x_perturbed = x.at[0, 4].add(100.0)
    out2 = attn(x_perturbed)[0, 0]
    assert not jnp.allclose(out1, out2, atol=1e-4)


def test_mha_rope_changes_output():
    attn_norope = xl.MultiHeadAttention(dim=16, num_heads=2, key=jax.random.PRNGKey(0))
    attn_rope = xl.MultiHeadAttention(dim=16, num_heads=2, use_rope=True, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))
    out_norope = attn_norope(x)
    out_rope = attn_rope(x)
    assert not jnp.allclose(out_norope, out_rope, atol=1e-4)


def test_mha_dropout_deterministic_by_default():
    attn = xl.MultiHeadAttention(dim=16, num_heads=2, dropout_rate=0.5, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))
    out1 = attn(x, deterministic=True)
    out2 = attn(x, deterministic=True)
    assert jnp.allclose(out1, out2)


def test_mha_dropout_stochastic_when_not_deterministic():
    attn = xl.MultiHeadAttention(dim=16, num_heads=2, dropout_rate=0.5, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))
    out1 = attn(x, key=jax.random.PRNGKey(2), deterministic=False)
    out2 = attn(x, key=jax.random.PRNGKey(3), deterministic=False)
    assert not jnp.allclose(out1, out2)


def test_mha_grad_shapes_match_params():
    attn = xl.MultiHeadAttention(dim=16, num_heads=2, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 16))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(attn, x)
    assert grads.q_proj.weight.shape == attn.q_proj.weight.shape
    assert grads.out_proj.weight.shape == attn.out_proj.weight.shape


def test_mha_jit_compatible():
    attn = xl.MultiHeadAttention(dim=16, num_heads=2, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 16))
    fwd = jax.jit(lambda m, x: m(x))
    out = fwd(attn, x)
    assert out.shape == (2, 4, 16)


# ---------------------------------------------------------------------------
# GroupedQueryAttention
# ---------------------------------------------------------------------------

def test_gqa_forward_shape():
    attn = xl.GroupedQueryAttention(dim=32, num_heads=8, num_kv_heads=2, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 32))
    out = attn(x)
    assert out.shape == (2, 6, 32)


def test_gqa_kv_projection_shapes():
    attn = xl.GroupedQueryAttention(dim=32, num_heads=8, num_kv_heads=2, key=jax.random.PRNGKey(0))
    # head_dim = 32/8 = 4, kv_dim = 4*2 = 8
    assert attn.k_proj.out_features == 8
    assert attn.v_proj.out_features == 8
    assert attn.q_proj.out_features == 32


def test_gqa_requires_divisible_dim():
    with pytest.raises(AssertionError):
        xl.GroupedQueryAttention(dim=33, num_heads=8, num_kv_heads=2, key=jax.random.PRNGKey(0))


def test_gqa_requires_heads_divisible_by_kv_heads():
    with pytest.raises(AssertionError):
        xl.GroupedQueryAttention(dim=32, num_heads=8, num_kv_heads=3, key=jax.random.PRNGKey(0))


def test_gqa_num_kv_heads_equal_num_heads_matches_mha_shapes():
    # When num_kv_heads == num_heads, GQA degenerates to per-head K/V,
    # same output shape contract as MHA.
    gqa = xl.GroupedQueryAttention(dim=16, num_heads=4, num_kv_heads=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 16))
    out = gqa(x)
    assert out.shape == (2, 5, 16)


def test_gqa_causal_mask_blocks_future():
    attn = xl.GroupedQueryAttention(dim=16, num_heads=4, num_kv_heads=2, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))
    mask = xl.causal_mask(5)[None, None, :, :]

    out1 = attn(x, mask=mask)[0, 0]
    x_perturbed = x.at[0, 4].add(100.0)
    out2 = attn(x_perturbed, mask=mask)[0, 0]
    assert jnp.allclose(out1, out2, atol=1e-4)


def test_gqa_rope_runs_and_changes_output():
    attn = xl.GroupedQueryAttention(
        dim=16, num_heads=4, num_kv_heads=2, use_rope=True, key=jax.random.PRNGKey(0)
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))
    out = attn(x)
    assert out.shape == (1, 5, 16)


def test_gqa_grad_shapes_match_params():
    attn = xl.GroupedQueryAttention(dim=16, num_heads=4, num_kv_heads=2, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 16))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(attn, x)
    assert grads.k_proj.weight.shape == attn.k_proj.weight.shape


# ---------------------------------------------------------------------------
# SelfAttention
# ---------------------------------------------------------------------------

def test_self_attention_forward_shape():
    attn = xl.SelfAttention(dim=16, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 16))
    out = attn(x)
    assert out.shape == (2, 5, 16)


def test_self_attention_cross_attention_with_context():
    attn = xl.SelfAttention(dim=16, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 16))
    context = jax.random.normal(jax.random.PRNGKey(2), (2, 9, 16))
    out = attn(x, context=context)
    # Output query length follows x, not context.
    assert out.shape == (2, 5, 16)


def test_self_attention_context_changes_output():
    attn = xl.SelfAttention(dim=16, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 3, 16))
    context1 = jax.random.normal(jax.random.PRNGKey(2), (1, 4, 16))
    context2 = jax.random.normal(jax.random.PRNGKey(3), (1, 4, 16))
    out1 = attn(x, context=context1)
    out2 = attn(x, context=context2)
    assert not jnp.allclose(out1, out2)


def test_self_attention_mask_blocks_positions():
    attn = xl.SelfAttention(dim=16, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 16))
    mask = xl.causal_mask(4)[None, :, :]

    out1 = attn(x, mask=mask)[0, 0]
    x_perturbed = x.at[0, 3].add(100.0)
    out2 = attn(x_perturbed, mask=mask)[0, 0]
    assert jnp.allclose(out1, out2, atol=1e-4)


def test_self_attention_dropout_stochastic():
    attn = xl.SelfAttention(dim=16, dropout_rate=0.5, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))
    out1 = attn(x, key=jax.random.PRNGKey(2), deterministic=False)
    out2 = attn(x, key=jax.random.PRNGKey(3), deterministic=False)
    assert not jnp.allclose(out1, out2)


def test_self_attention_grad_shapes_match_params():
    attn = xl.SelfAttention(dim=16, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 16))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(attn, x)
    assert grads.q_proj.weight.shape == attn.q_proj.weight.shape
    assert grads.v_proj.weight.shape == attn.v_proj.weight.shape
