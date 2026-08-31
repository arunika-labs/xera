"""Tests for xera.loom.initializers: lecun, xavier, kaiming, orthogonal, variance_scaling, etc."""

import jax
import jax.numpy as jnp
import pytest
from xera.loom import initializers as init


SHAPES_2D = [(8, 16), (16, 8)]


@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("init_fn", [
    init.lecun_normal(),
    init.xavier_normal(),
    init.xavier_uniform(),
    init.kaiming_normal(),
    init.kaiming_uniform(),
    init.normal(),
    init.uniform(),
    init.truncated_normal(),
])
def test_initializer_shape_and_dtype(init_fn, shape):
    key = jax.random.PRNGKey(0)
    out = init_fn(key, shape, jnp.float32)
    assert out.shape == shape
    assert out.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(out))


def test_zeros_and_ones():
    key = jax.random.PRNGKey(0)
    shape = (4, 4)
    assert jnp.all(init.zeros()(key, shape) == 0.0)
    assert jnp.all(init.ones()(key, shape) == 1.0)


def test_constant():
    key = jax.random.PRNGKey(0)
    out = init.constant(3.5)(key, (2, 3))
    assert jnp.all(out == 3.5)


def test_deterministic_given_same_key():
    key = jax.random.PRNGKey(0)
    shape = (8, 8)
    a = init.xavier_normal()(key, shape)
    b = init.xavier_normal()(key, shape)
    assert jnp.array_equal(a, b)


def test_different_keys_give_different_values():
    shape = (8, 8)
    a = init.kaiming_normal()(jax.random.PRNGKey(0), shape)
    b = init.kaiming_normal()(jax.random.PRNGKey(1), shape)
    assert not jnp.array_equal(a, b)


def test_uniform_respects_scale_bound():
    key = jax.random.PRNGKey(0)
    out = init.uniform(scale=0.1)(key, (1000,))
    assert jnp.all(out >= -0.1) and jnp.all(out <= 0.1)


def test_lecun_normal_std_scales_with_fan_in():
    # std should shrink as fan_in grows: 1/sqrt(fan_in)
    key = jax.random.PRNGKey(0)
    small = init.lecun_normal()(key, (10000, 4))
    large = init.lecun_normal()(key, (10000, 400))
    assert jnp.std(small) > jnp.std(large)


def test_xavier_normal_uses_average_fan():
    key = jax.random.PRNGKey(0)
    out = init.xavier_normal()(key, (10000, 4))
    fan_in, fan_out = 10000, 4
    expected_std = (2.0 / (fan_in + fan_out)) ** 0.5
    assert jnp.isclose(jnp.std(out), expected_std, rtol=0.1)


def test_kaiming_normal_uses_fan_in_only():
    key = jax.random.PRNGKey(0)
    fan_in = 256
    out = init.kaiming_normal()(key, (fan_in, 4))
    expected_std = (2.0 / fan_in) ** 0.5
    assert jnp.isclose(jnp.std(out), expected_std, rtol=0.15)


@pytest.mark.parametrize("shape", [(8, 8), (16, 4), (4, 16)])
def test_orthogonal_rows_or_cols_are_orthonormal(shape):
    key = jax.random.PRNGKey(0)
    q = init.orthogonal(scale=1.0)(key, shape, jnp.float32)
    assert q.shape == shape
    n_rows, n_cols = shape
    if n_rows >= n_cols:
        gram = q.T @ q
        assert jnp.allclose(gram, jnp.eye(n_cols), atol=1e-4)
    else:
        gram = q @ q.T
        assert jnp.allclose(gram, jnp.eye(n_rows), atol=1e-4)


def test_orthogonal_scale_multiplies_output():
    key = jax.random.PRNGKey(0)
    shape = (8, 8)
    q1 = init.orthogonal(scale=1.0)(key, shape, jnp.float32)
    q2 = init.orthogonal(scale=2.0)(key, shape, jnp.float32)
    assert jnp.allclose(q2, q1 * 2.0, atol=1e-5)


def test_orthogonal_rejects_1d_shape():
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError):
        init.orthogonal()(key, (8,), jnp.float32)


@pytest.mark.parametrize("mode", ["fan_in", "fan_out", "fan_avg"])
@pytest.mark.parametrize("distribution", ["normal", "truncated_normal", "uniform"])
def test_variance_scaling_all_modes_and_distributions(mode, distribution):
    key = jax.random.PRNGKey(0)
    fn = init.variance_scaling(scale=1.0, mode=mode, distribution=distribution)
    out = fn(key, (32, 16), jnp.float32)
    assert out.shape == (32, 16)
    assert jnp.all(jnp.isfinite(out))


def test_variance_scaling_unknown_mode_raises():
    fn = init.variance_scaling(mode="bogus")
    with pytest.raises(ValueError):
        fn(jax.random.PRNGKey(0), (4, 4))


def test_variance_scaling_unknown_distribution_raises():
    fn = init.variance_scaling(distribution="bogus")
    with pytest.raises(ValueError):
        fn(jax.random.PRNGKey(0), (4, 4))


def test_fan_in_out_for_conv_like_shape():
    # 4D conv kernel shape: (kh, kw, in_ch, out_ch)
    from xera.loom.initializers import _fan_in_out

    fan_in, fan_out = _fan_in_out((3, 3, 8, 16))
    receptive_field = 3 * 3
    assert fan_in == 8 * receptive_field
    assert fan_out == 16 * receptive_field


def test_fan_in_out_for_1d_shape():
    from xera.loom.initializers import _fan_in_out

    fan_in, fan_out = _fan_in_out((32,))
    assert fan_in == fan_out == 32
