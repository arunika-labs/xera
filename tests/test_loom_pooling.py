"""Tests for xera.xl.pooling: MaxPool, AvgPool, GlobalAvgPool."""

import jax
import jax.numpy as jnp
import xera.loom as xl


def test_pooling_shapes():
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8, 3))
    assert xl.MaxPool(pool_size=(2, 2))(x).shape == (2, 4, 4, 3)
    assert xl.AvgPool(pool_size=(2, 2))(x).shape == (2, 4, 4, 3)
    assert xl.GlobalAvgPool()(x).shape == (2, 3)
    assert xl.GlobalAvgPool(keepdims=True)(x).shape == (2, 1, 1, 3)


def test_max_pool_picks_max_value():
    x = jnp.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])  # (1, 2, 2, 1)
    out = xl.MaxPool(pool_size=(2, 2))(x)
    assert out.shape == (1, 1, 1, 1)
    assert jnp.isclose(out[0, 0, 0, 0], 4.0)


def test_avg_pool_averages_value():
    x = jnp.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])  # (1, 2, 2, 1)
    out = xl.AvgPool(pool_size=(2, 2))(x)
    assert jnp.isclose(out[0, 0, 0, 0], 2.5)


def test_global_avg_pool_matches_mean():
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 4, 3))
    out = xl.GlobalAvgPool()(x)
    expected = jnp.mean(x, axis=(1, 2))
    assert jnp.allclose(out, expected)
