"""Tests for xera.loom.stochastic: Dropout."""

import jax
import jax.numpy as jnp
import xera.loom as loom


def test_dropout_deterministic_is_identity():
    dropout = loom.Dropout(rate=0.5)
    x = jax.random.normal(jax.random.PRNGKey(0), (100,))
    out = dropout(x, key=jax.random.PRNGKey(1), deterministic=True)
    assert jnp.array_equal(out, x)


def test_dropout_deterministic_default_true():
    # Default (no kwargs at all) must be eval mode -> identity.
    dropout = loom.Dropout(rate=0.9)
    x = jax.random.normal(jax.random.PRNGKey(0), (50,))
    out = dropout(x)
    assert jnp.array_equal(out, x)


def test_dropout_rate_zero_is_identity_even_when_training():
    dropout = loom.Dropout(rate=0.0)
    x = jax.random.normal(jax.random.PRNGKey(0), (50,))
    out = dropout(x, key=jax.random.PRNGKey(1), deterministic=False)
    assert jnp.array_equal(out, x)


def test_dropout_training_zeros_out_some_units():
    dropout = loom.Dropout(rate=0.5)
    x = jnp.ones((1000,))
    out = dropout(x, key=jax.random.PRNGKey(0), deterministic=False)
    frac_zero = jnp.mean(out == 0.0)
    # roughly rate fraction should be zeroed (allow generous tolerance)
    assert 0.35 < frac_zero < 0.65


def test_dropout_training_scales_kept_units_by_inverse_keep_prob():
    dropout = loom.Dropout(rate=0.5)
    x = jnp.ones((1000,))
    out = dropout(x, key=jax.random.PRNGKey(0), deterministic=False)
    kept = out[out != 0.0]
    assert jnp.allclose(kept, 2.0)  # 1 / (1 - 0.5) = 2.0


def test_dropout_preserves_expected_value_roughly():
    # E[dropout(x)] ~= x, since kept units are scaled by 1/keep_prob
    dropout = loom.Dropout(rate=0.3)
    x = jnp.ones((5000,))
    out = dropout(x, key=jax.random.PRNGKey(0), deterministic=False)
    assert jnp.isclose(jnp.mean(out), 1.0, atol=0.05)


def test_dropout_different_keys_give_different_masks():
    dropout = loom.Dropout(rate=0.5)
    x = jnp.ones((200,))
    out1 = dropout(x, key=jax.random.PRNGKey(0), deterministic=False)
    out2 = dropout(x, key=jax.random.PRNGKey(1), deterministic=False)
    assert not jnp.array_equal(out1, out2)
