"""Tests for xera.loom.functional: activation functions re-exported from jax.nn."""

import jax
import jax.nn as jnn
import jax.numpy as jnp
import pytest
import xera.loom as loom


ACTIVATION_NAMES = [
    "celu",
    "elu",
    "gelu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "logsumexp",
    "mish",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "soft_sign",
    "softmax",
    "softplus",
    "squareplus",
    "standardize",
    "swish",
    "tanh",
]


def test_all_activations_exposed_on_loom():
    for name in ACTIVATION_NAMES:
        assert hasattr(loom, name), f"xera.loom is missing '{name}'"


def test_glu_and_one_hot_exposed_on_loom():
    assert hasattr(loom, "glu")
    assert hasattr(loom, "one_hot")


@pytest.mark.parametrize("name", ACTIVATION_NAMES)
def test_activation_is_identical_to_jax_nn(name):
    # These are re-exports, not reimplementations: identity check, not just
    # numerical closeness.
    assert getattr(loom, name) is getattr(jnn, name)


@pytest.mark.parametrize("name", ACTIVATION_NAMES)
def test_activation_matches_jax_nn_output(name):
    x = jax.random.normal(jax.random.PRNGKey(0), (16,))
    loom_fn = getattr(loom, name)
    jnn_fn = getattr(jnn, name)
    assert jnp.array_equal(loom_fn(x), jnn_fn(x))


def test_relu_zeroes_negatives():
    x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    out = loom.relu(x)
    assert jnp.array_equal(out, jnp.array([0.0, 0.0, 0.0, 0.5, 2.0]))


def test_sigmoid_output_range():
    x = jax.random.normal(jax.random.PRNGKey(0), (100,))
    out = loom.sigmoid(x)
    assert jnp.all(out > 0.0) and jnp.all(out < 1.0)


def test_softmax_sums_to_one():
    x = jax.random.normal(jax.random.PRNGKey(0), (10,))
    out = loom.softmax(x)
    assert jnp.isclose(jnp.sum(out), 1.0, atol=1e-5)


def test_tanh_output_range():
    x = jax.random.normal(jax.random.PRNGKey(0), (100,))
    out = loom.tanh(x)
    assert jnp.all(out > -1.0) and jnp.all(out < 1.0)


def test_one_hot_basic():
    labels = jnp.array([0, 2, 1])
    out = loom.one_hot(labels, num_classes=3)
    expected = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert jnp.array_equal(out, expected)


def test_functional_all_matches_loom_reexports():
    from xera.loom import functional

    for name in functional.__all__:
        assert hasattr(loom, name), f"'{name}' in functional.__all__ but not exported from xera.loom"
