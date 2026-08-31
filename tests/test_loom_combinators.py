"""Tests for xera.xl.combinators: Sequential, Residual, Lambda.

The headline regression here: Sequential used to hardcode
`if isinstance(layer, Dropout): x = layer(x, **kwargs)` for forwarding
kwargs, which meant a BatchNorm placed inside Sequential could never
receive `deterministic=` -- there was no way to put it into eval mode, so
its running stats would silently keep updating during inference. Sequential
now inspects each layer's `__call__` signature and forwards only the
kwargs that layer declares, and threads through stateful (tuple-returning)
layers like BatchNorm automatically.
"""

import jax
import jax.numpy as jnp
import xera.loom as xl


# ---------------------------------------------------------------------------
# Sequential: plain (no kwargs / no stateful layers)
# ---------------------------------------------------------------------------

def test_sequential_forward_shape():
    model = xl.Sequential([
        xl.Dense(4, 8, key=jax.random.PRNGKey(0)),
        xl.Dense(8, 2, key=jax.random.PRNGKey(1)),
    ])
    x = jax.random.normal(jax.random.PRNGKey(2), (3, 4))
    out = model(x)
    assert out.shape == (3, 2)


def test_sequential_applies_layers_in_order():
    # Dense with zero weight/bias except an offset makes ordering checkable
    d1 = xl.Dense(2, 2, key=jax.random.PRNGKey(0))
    d2 = xl.Dense(2, 2, key=jax.random.PRNGKey(1))
    model = xl.Sequential([d1, d2])
    x = jax.random.normal(jax.random.PRNGKey(2), (1, 2))
    assert jnp.allclose(model(x), d2(d1(x)))


# ---------------------------------------------------------------------------
# Sequential + Dropout: kwargs forwarded to layers that declare them
# ---------------------------------------------------------------------------

def test_sequential_forwards_deterministic_and_key_to_dropout():
    model = xl.Sequential([
        xl.Dense(4, 4, key=jax.random.PRNGKey(0)),
        xl.Dropout(rate=0.9),
    ])
    x = jnp.ones((100, 4))

    out_eval = model(x, deterministic=True)
    out_train = model(x, key=jax.random.PRNGKey(1), deterministic=False)

    # eval mode: dropout is a no-op, so nothing should be zeroed by it
    assert not jnp.any(out_eval == 0.0)
    # train mode with rate=0.9: most units should be dropped
    assert jnp.mean(out_train == 0.0) > 0.5


def test_sequential_dense_only_ignores_unrelated_kwargs():
    # Dense's __call__ doesn't take `deterministic` -- Sequential must not
    # error out just because the kwarg was passed for other layers.
    model = xl.Sequential([xl.Dense(4, 4, key=jax.random.PRNGKey(0))])
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4))
    out = model(x, deterministic=True, key=jax.random.PRNGKey(2))
    assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# Sequential + BatchNorm: the actual bug regression
# ---------------------------------------------------------------------------

def test_sequential_forwards_deterministic_to_batchnorm():
    # This is the reported bug: BatchNorm inside Sequential must be
    # toggleable to eval mode via `deterministic=True`.
    bn = xl.BatchNorm(dim=4, key=jax.random.PRNGKey(0))
    model = xl.Sequential([bn])
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4))

    # eval mode (default / explicit deterministic=True): running stats must
    # NOT change. In eval mode BatchNorm returns itself unchanged, so
    # Sequential detects nothing stateful happened and returns just the array.
    out = model(x, deterministic=True)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (8, 4)


def test_sequential_default_kwargs_puts_batchnorm_in_eval_mode():
    # Calling Sequential with no kwargs at all must also leave BatchNorm's
    # running stats untouched (deterministic=True is BatchNorm's default).
    bn = xl.BatchNorm(dim=4, key=jax.random.PRNGKey(0))
    model = xl.Sequential([bn])
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4))
    out = model(x)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (8, 4)


def test_sequential_deterministic_false_updates_batchnorm_running_stats():
    bn = xl.BatchNorm(dim=4, key=jax.random.PRNGKey(0))
    model = xl.Sequential([bn])
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4))

    out, new_model = model(x, deterministic=False)
    assert not jnp.allclose(
        model.layers[0].running_mean.value,
        new_model.layers[0].running_mean.value,
    )
    assert out.shape == (8, 4)


def test_sequential_mixed_dense_batchnorm_dropout_eval_mode():
    # A realistic stack: Dense -> BatchNorm -> Dropout, all toggled to eval
    # via a single deterministic=True kwarg, exactly like the bug report.
    model = xl.Sequential([
        xl.Dense(4, 4, key=jax.random.PRNGKey(0)),
        xl.BatchNorm(dim=4, key=jax.random.PRNGKey(1)),
        xl.Dropout(rate=0.5),
    ])
    x = jax.random.normal(jax.random.PRNGKey(2), (8, 4))

    # In eval mode, BatchNorm doesn't produce updated state, so Sequential
    # stays backward compatible and returns just the output array.
    out = model(x, deterministic=True)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (8, 4)


def test_sequential_mixed_dense_batchnorm_dropout_train_mode():
    model = xl.Sequential([
        xl.Dense(4, 4, key=jax.random.PRNGKey(0)),
        xl.BatchNorm(dim=4, key=jax.random.PRNGKey(1)),
        xl.Dropout(rate=0.5),
    ])
    x = jax.random.normal(jax.random.PRNGKey(2), (8, 4))

    out, new_model = model(x, key=jax.random.PRNGKey(3), deterministic=False)
    assert out.shape == (8, 4)
    bn_before = model.layers[1]
    bn_after = new_model.layers[1]
    # running stats must actually update in training mode
    assert not jnp.allclose(bn_before.running_mean.value, bn_after.running_mean.value)


def test_sequential_without_stateful_layers_returns_plain_output():
    # No BatchNorm-like layer in the stack -> Sequential should return just
    # the tensor, not a (output, new_self) tuple, to stay backward
    # compatible with plain Dense-only stacks.
    model = xl.Sequential([
        xl.Dense(4, 4, key=jax.random.PRNGKey(0)),
        xl.Dropout(rate=0.0),
    ])
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4))
    out = model(x, deterministic=True)
    assert isinstance(out, jnp.ndarray)


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------

def test_residual_adds_input_to_inner_output():
    inner = xl.Dense(4, 4, key=jax.random.PRNGKey(0))
    block = xl.Residual(inner)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4))
    out = block(x)
    assert jnp.allclose(out, x + inner(x))


def test_residual_forwards_kwargs_to_dropout():
    block = xl.Residual(xl.Dropout(rate=0.9))
    x = jnp.ones((200, 4))
    out_eval = block(x, deterministic=True)
    assert jnp.allclose(out_eval, x + x)  # dropout no-op in eval -> x + x

    out_train = block(x, key=jax.random.PRNGKey(0), deterministic=False)
    assert not jnp.allclose(out_train, x + x)


def test_residual_handles_stateful_inner_batchnorm():
    bn = xl.BatchNorm(dim=4, key=jax.random.PRNGKey(0))
    block = xl.Residual(bn)
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4))

    result = block(x, deterministic=False)
    assert isinstance(result, tuple)
    out, new_block = result
    assert out.shape == (8, 4)
    assert not jnp.allclose(
        block.inner.running_mean.value, new_block.inner.running_mean.value
    )


# ---------------------------------------------------------------------------
# Lambda
# ---------------------------------------------------------------------------

def test_lambda_applies_wrapped_function():
    relu = xl.Lambda(lambda x: jnp.maximum(0, x))
    x = jnp.array([-1.0, 0.0, 2.0])
    out = relu(x)
    assert jnp.array_equal(out, jnp.array([0.0, 0.0, 2.0]))


def test_lambda_ignores_extra_kwargs():
    identity = xl.Lambda(lambda x: x)
    x = jnp.array([1.0, 2.0])
    out = identity(x, deterministic=True, key=jax.random.PRNGKey(0))
    assert jnp.array_equal(out, x)


def test_sequential_with_lambda_and_dense():
    model = xl.Sequential([
        xl.Dense(4, 4, key=jax.random.PRNGKey(0)),
        xl.Lambda(lambda x: jax.nn.relu(x)),
    ])
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4))
    out = model(x)
    assert jnp.all(out >= 0.0)
