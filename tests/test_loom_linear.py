"""Tests for xera.xl.linear: Dense."""

import jax
import jax.numpy as jnp
import xera.loom as xl


def test_dense_forward_shape():
    dense = xl.Dense(4, 8, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4))
    out = dense(x)
    assert out.shape == (2, 8)


def test_dense_no_bias():
    dense = xl.Dense(4, 8, use_bias=False, key=jax.random.PRNGKey(0))
    assert dense.bias is None
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4))
    out = dense(x)
    assert out.shape == (2, 8)


def test_dense_bias_defaults_to_zero():
    dense = xl.Dense(4, 8, key=jax.random.PRNGKey(0))
    assert jnp.allclose(dense.bias, jnp.zeros(8))


def test_dense_grad_shapes_match_params():
    dense = xl.Dense(4, 4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(dense, x)
    assert grads.weight.shape == dense.weight.shape
    assert grads.bias.shape == dense.bias.shape


def test_dense_batched_input():
    dense = xl.Dense(4, 8, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 4))
    out = dense(x)
    assert out.shape == (2, 5, 8)
