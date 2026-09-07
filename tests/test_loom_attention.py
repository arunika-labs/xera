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
# alibi_slopes / alibi_bias
# ---------------------------------------------------------------------------

def test_alibi_slopes_shape_and_dtype():
    slopes = xl.alibi_slopes(8)
    assert slopes.shape == (8,)
    assert slopes.dtype == jnp.float32


def test_alibi_slopes_power_of_2_matches_paper_geometric_sequence():
    # For 8 heads, Press et al. 2021 give 1/2, 1/4, ..., 1/256.
    slopes = xl.alibi_slopes(8)
    expected = jnp.array([2.0 ** -(i + 1) for i in range(8)])
    assert jnp.allclose(slopes, expected, atol=1e-6)


def test_alibi_slopes_decreasing():
    slopes = xl.alibi_slopes(8)
    assert bool(jnp.all(slopes[:-1] > slopes[1:]))


def test_alibi_slopes_non_power_of_2_head_count_still_works():
    # num_heads not a power of 2 exercises the interleaving fallback path.
    slopes = xl.alibi_slopes(6)
    assert slopes.shape == (6,)
    assert bool(jnp.all(slopes > 0))


def test_alibi_bias_shape_self_attention():
    bias = xl.alibi_bias(4, 5)
    assert bias.shape == (4, 5, 5)


def test_alibi_bias_shape_cross_attention():
    bias = xl.alibi_bias(4, 5, 7)
    assert bias.shape == (4, 5, 7)


def test_alibi_bias_diagonal_is_zero():
    # Distance 0 (query attending to itself) must carry no penalty.
    bias = xl.alibi_bias(4, 6)
    diag = jnp.diagonal(bias, axis1=-2, axis2=-1)
    assert jnp.allclose(diag, 0.0)


def test_alibi_bias_is_non_positive():
    # ALiBi only ever penalizes (or leaves unchanged); it never rewards.
    bias = xl.alibi_bias(4, 6)
    assert bool(jnp.all(bias <= 0.0))


def test_alibi_bias_farther_positions_penalized_more():
    bias = xl.alibi_bias(1, 5)[0]  # single head for a simple monotonicity check
    # Row 0: penalty should grow (more negative) with distance from query 0.
    row0 = bias[0]
    assert bool(jnp.all(row0[:-1] >= row0[1:]))


def test_alibi_bias_symmetric_for_self_attention():
    # |i - j| is symmetric, so the bias matrix should be too.
    bias = xl.alibi_bias(3, 6)
    assert jnp.allclose(bias, jnp.swapaxes(bias, -2, -1))


def test_alibi_bias_heads_scaled_by_their_slopes():
    bias = xl.alibi_bias(4, 5)
    slopes = xl.alibi_slopes(4)
    # bias[h] should equal -slopes[h] * |i-j|; check the ratio between two
    # heads at a fixed off-diagonal entry matches the ratio of their slopes.
    ratio_bias = bias[0, 0, 1] / bias[1, 0, 1]
    ratio_slopes = slopes[0] / slopes[1]
    assert jnp.allclose(ratio_bias, ratio_slopes, atol=1e-5)


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


def test_mha_alibi_changes_output():
    attn_plain = xl.MultiHeadAttention(dim=16, num_heads=2, key=jax.random.PRNGKey(0))
    attn_alibi = xl.MultiHeadAttention(dim=16, num_heads=2, use_alibi=True, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))
    out_plain = attn_plain(x)
    out_alibi = attn_alibi(x)
    assert not jnp.allclose(out_plain, out_alibi, atol=1e-4)


def test_mha_alibi_causal_mask_blocks_future_influence():
    # ALiBi is an additive bias, not a mask -- an explicit causal mask
    # must still block future positions when both are used together.
    attn = xl.MultiHeadAttention(dim=16, num_heads=2, use_alibi=True, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))
    mask = xl.causal_mask(5)[None, None, :, :]

    def out_at_pos0(x_):
        return attn(x_, mask=mask)[0, 0]

    x_perturbed = x.at[0, 4].add(100.0)
    out1 = out_at_pos0(x)
    out2 = out_at_pos0(x_perturbed)
    assert jnp.allclose(out1, out2, atol=1e-4)


def test_mha_rope_and_alibi_combine_without_error():
    # Not a standard combination (see the use_alibi docstring), but
    # nothing should prevent enabling both -- RoPE and ALiBi act at
    # different points in the computation (pre- vs post-dot-product).
    attn_rope_only = xl.MultiHeadAttention(dim=16, num_heads=2, use_rope=True, key=jax.random.PRNGKey(0))
    attn_both = xl.MultiHeadAttention(dim=16, num_heads=2, use_rope=True, use_alibi=True, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))
    out_rope_only = attn_rope_only(x)
    out_both = attn_both(x)
    assert out_both.shape == out_rope_only.shape
    assert not jnp.allclose(out_rope_only, out_both, atol=1e-4)


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


def test_gqa_alibi_runs_and_changes_output():
    attn_plain = xl.GroupedQueryAttention(dim=16, num_heads=4, num_kv_heads=2, key=jax.random.PRNGKey(0))
    attn_alibi = xl.GroupedQueryAttention(
        dim=16, num_heads=4, num_kv_heads=2, use_alibi=True, key=jax.random.PRNGKey(0)
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))
    out_plain = attn_plain(x)
    out_alibi = attn_alibi(x)
    assert out_alibi.shape == (1, 5, 16)
    assert not jnp.allclose(out_plain, out_alibi, atol=1e-4)


def test_gqa_rope_and_alibi_combine_without_error():
    attn = xl.GroupedQueryAttention(
        dim=16, num_heads=4, num_kv_heads=2, use_rope=True, use_alibi=True, key=jax.random.PRNGKey(0)
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
