"""Tests for xera.loom.normalization: LayerNorm, RMSNorm, BatchNorm, GroupNorm,
InstanceNorm, LayerScale, GroupNormWithRunningStats.

BatchNorm and GroupNormWithRunningStats use `deterministic` (default True,
i.e. eval mode) to match every other stochastic/stateful layer in the
framework (Dropout, MLP, TransformerBlock, attention variants). This is a
deliberate polarity flip from the old `training=True` kwarg -- see the
regression tests below.
"""

import jax
import jax.numpy as jnp
import pytest
import xera.loom as loom


def test_layer_norm_normalizes_last_axis():
    ln = loom.LayerNorm(dim=8, key=jax.random.PRNGKey(99))
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 8)) * 5 + 3
    out = ln(x)
    assert out.shape == x.shape
    assert jnp.allclose(jnp.mean(out, axis=-1), 0.0, atol=1e-5)
    assert jnp.allclose(jnp.std(out, axis=-1), 1.0, atol=1e-3)


def test_rms_norm_shape_and_no_mean_centering():
    rms = loom.RMSNorm(dim=8, key=jax.random.PRNGKey(99))
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 8)) + 10.0  # nonzero mean
    out = rms(x)
    assert out.shape == x.shape
    # RMSNorm does NOT center the mean, unlike LayerNorm
    assert not jnp.allclose(jnp.mean(out, axis=-1), 0.0, atol=1e-2)


def test_group_norm_shape():
    gn = loom.GroupNorm(num_groups=4, dim=16, key=jax.random.PRNGKey(99))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8, 16))
    out = gn(x)
    assert out.shape == x.shape


def test_instance_norm_normalizes_per_sample_spatial():
    inorm = loom.InstanceNorm(dim=4, key=jax.random.PRNGKey(99))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 5, 4)) * 3 + 1
    out = inorm(x)
    assert out.shape == x.shape
    mean_per_sample = jnp.mean(out, axis=(1, 2))
    assert jnp.allclose(mean_per_sample, 0.0, atol=1e-4)


def test_layer_scale_scales_by_learnable_init_value():
    ls = loom.LayerScale(dim=4, init_value=0.5, key=jax.random.PRNGKey(99))
    x = jnp.ones((2, 4))
    out = ls(x)
    assert jnp.allclose(out, 0.5)


# ---------------------------------------------------------------------------
# BatchNorm: `deterministic` kwarg (regression coverage for the training=
# -> deterministic= rename, and the polarity flip default=True == eval)
# ---------------------------------------------------------------------------

def test_batchnorm_default_is_eval_mode():
    # Default (no kwarg) must be eval mode: uses running stats, does not
    # update them. This matches Dropout/MLP/etc's deterministic=True default.
    bn = loom.BatchNorm(dim=16, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 16))
    out, new_bn = bn(x)
    assert jnp.allclose(bn.running_mean.value, new_bn.running_mean.value)
    assert jnp.allclose(bn.running_var.value, new_bn.running_var.value)


def test_batchnorm_deterministic_false_updates_running_stats():
    bn = loom.BatchNorm(dim=16, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 16))
    _, bn2 = bn(x, deterministic=False)
    assert not jnp.allclose(bn.running_mean.value, bn2.running_mean.value)
    assert jnp.allclose(bn.gamma, bn2.gamma)
    assert jnp.allclose(bn.beta, bn2.beta)


def test_batchnorm_no_longer_accepts_training_kwarg():
    # `training=` was the old, inconsistent kwarg name -- confirm it's gone
    # so nobody silently falls back to a stale default.
    bn = loom.BatchNorm(dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4))
    with pytest.raises(TypeError):
        bn(x, training=True)


def test_batchnorm_state_separate_from_params():
    bn = loom.BatchNorm(dim=16, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 16))
    _, bn2 = bn(x, deterministic=False)
    assert not jnp.allclose(bn.running_mean.value, bn2.running_mean.value)
    assert jnp.allclose(bn.gamma, bn2.gamma)
    assert jnp.allclose(bn.beta, bn2.beta)


def test_batchnorm_eval_uses_running_stats_not_batch_stats():
    bn = loom.BatchNorm(dim=4, momentum=0.0, key=jax.random.PRNGKey(0))
    x_train = jax.random.normal(jax.random.PRNGKey(1), (16, 4)) * 3 + 5
    _, bn_trained = bn(x_train, deterministic=False)  # running stats = batch stats (momentum=0)

    x_eval = jax.random.normal(jax.random.PRNGKey(2), (16, 4)) * 0.01  # near-zero, different stats
    out_eval, bn_after_eval = bn_trained(x_eval, deterministic=True)

    # eval mode must not have changed running stats
    assert jnp.allclose(bn_trained.running_mean.value, bn_after_eval.running_mean.value)
    # and normalization should be using the *training* batch's stats, not x_eval's
    # (i.e. output should not itself be ~zero-mean/unit-var for x_eval)
    assert not jnp.allclose(jnp.mean(out_eval, axis=0), 0.0, atol=0.5)


# ---------------------------------------------------------------------------
# GroupNormWithRunningStats: same deterministic= contract as BatchNorm
# ---------------------------------------------------------------------------

def test_group_norm_running_stats_default_is_eval_mode():
    gn = loom.GroupNormWithRunningStats(num_groups=4, dim=16, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 6, 16))
    out, new_gn = gn(x)
    assert jnp.allclose(gn.running_mean.value, new_gn.running_mean.value)


def test_group_norm_running_stats_deterministic_false_updates_stats():
    gn = loom.GroupNormWithRunningStats(num_groups=4, dim=16, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 6, 16))
    _, gn2 = gn(x, deterministic=False)
    assert not jnp.allclose(gn.running_mean.value, gn2.running_mean.value)


def test_group_norm_running_stats_no_longer_accepts_training_kwarg():
    gn = loom.GroupNormWithRunningStats(num_groups=2, dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4))
    with pytest.raises(TypeError):
        gn(x, training=True)
