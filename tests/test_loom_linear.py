"""Tests for xera.xl.linear: Linear, LoRALinear, DoRALinear."""

import jax
import jax.numpy as jnp
import pytest
import xera.loom as xl


def test_dense_forward_shape():
    linear = xl.Linear(4, 8, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4))
    out = linear(x)
    assert out.shape == (2, 8)


def test_dense_no_bias():
    linear = xl.Linear(4, 8, use_bias=False, key=jax.random.PRNGKey(0))
    assert linear.bias is None
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4))
    out = linear(x)
    assert out.shape == (2, 8)


def test_dense_bias_defaults_to_zero():
    linear = xl.Linear(4, 8, key=jax.random.PRNGKey(0))
    assert jnp.allclose(linear.bias, jnp.zeros(8))


def test_dense_grad_shapes_match_params():
    linear = xl.Linear(4, 4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(linear, x)
    assert grads.weight.shape == linear.weight.shape
    assert grads.bias.shape == linear.bias.shape


def test_dense_batched_input():
    linear = xl.Linear(4, 8, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 4))
    out = linear(x)
    assert out.shape == (2, 5, 8)


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

def test_lora_owned_base_forward_shape():
    base_w = jax.random.normal(jax.random.PRNGKey(2), (4, 8)) * 0.1
    lora = xl.LoRALinear(4, 8, rank=2, base_weight=base_w, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    out = lora(x)
    assert out.shape == (3, 8)


def test_lora_owned_base_is_noop_at_init():
    # lora_b starts at zero, so the adapted layer must match x @ base_weight
    # exactly before any training happens.
    base_w = jax.random.normal(jax.random.PRNGKey(2), (4, 8)) * 0.1
    lora = xl.LoRALinear(4, 8, rank=2, base_weight=base_w, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    assert jnp.allclose(lora(x), x @ base_w, atol=1e-5)


def test_lora_shared_base_is_noop_at_init():
    base_w = jax.random.normal(jax.random.PRNGKey(2), (4, 8)) * 0.1
    lora = xl.LoRALinear(4, 8, rank=2, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    assert jnp.allclose(lora(x, base_weight=base_w), x @ base_w, atol=1e-5)


def test_lora_raises_when_base_missing():
    lora = xl.LoRALinear(4, 8, rank=2, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    with pytest.raises(ValueError):
        lora(x)


def test_lora_raises_when_base_given_twice():
    base_w = jax.random.normal(jax.random.PRNGKey(2), (4, 8)) * 0.1
    lora = xl.LoRALinear(4, 8, rank=2, base_weight=base_w, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    with pytest.raises(ValueError):
        lora(x, base_weight=base_w)


def test_lora_base_weight_receives_no_gradient():
    # base_weight must be frozen (stop_gradient) regardless of whether it
    # sits inside a jax.grad call that also updates lora_a/lora_b.
    base_w = jax.random.normal(jax.random.PRNGKey(2), (4, 8)) * 0.1
    lora = xl.LoRALinear(4, 8, rank=2, base_weight=base_w, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(lora, x)
    assert jnp.allclose(grads.base_weight, 0.0)


def test_lora_grad_shapes_match_params():
    base_w = jax.random.normal(jax.random.PRNGKey(2), (4, 8)) * 0.1
    lora = xl.LoRALinear(4, 8, rank=2, base_weight=base_w, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(lora, x)
    assert grads.lora_a.shape == lora.lora_a.shape
    assert grads.lora_b.shape == lora.lora_b.shape


# ---------------------------------------------------------------------------
# DoRALinear
# ---------------------------------------------------------------------------

def test_dora_owned_base_forward_shape():
    base_w = jax.random.normal(jax.random.PRNGKey(2), (4, 8)) * 0.1
    dora = xl.DoRALinear(4, 8, rank=2, base_weight=base_w, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    out = dora(x)
    assert out.shape == (3, 8)


def test_dora_owned_base_is_noop_at_init():
    # lora_b starts at zero and magnitude is initialized from the base's
    # own column norms, so the effective weight at init equals base_weight
    # exactly (the column-norm normalization is cancelled by magnitude).
    base_w = jax.random.normal(jax.random.PRNGKey(2), (4, 8)) * 0.1
    dora = xl.DoRALinear(4, 8, rank=2, base_weight=base_w, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    assert jnp.allclose(dora(x), x @ base_w, atol=1e-4)


def test_dora_shared_base_forward_shape():
    base_w = jax.random.normal(jax.random.PRNGKey(2), (4, 8)) * 0.1
    dora = xl.DoRALinear(4, 8, rank=2, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    out = dora(x, base_weight=base_w)
    assert out.shape == (3, 8)


def test_dora_raises_when_base_missing():
    dora = xl.DoRALinear(4, 8, rank=2, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    with pytest.raises(ValueError):
        dora(x)


def test_dora_raises_when_base_given_twice():
    base_w = jax.random.normal(jax.random.PRNGKey(2), (4, 8)) * 0.1
    dora = xl.DoRALinear(4, 8, rank=2, base_weight=base_w, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    with pytest.raises(ValueError):
        dora(x, base_weight=base_w)


def test_dora_base_weight_receives_no_gradient():
    base_w = jax.random.normal(jax.random.PRNGKey(2), (4, 8)) * 0.1
    dora = xl.DoRALinear(4, 8, rank=2, base_weight=base_w, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(dora, x)
    assert jnp.allclose(grads.base_weight, 0.0)


def test_dora_grad_shapes_match_params():
    base_w = jax.random.normal(jax.random.PRNGKey(2), (4, 8)) * 0.1
    dora = xl.DoRALinear(4, 8, rank=2, base_weight=base_w, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(dora, x)
    assert grads.lora_a.shape == dora.lora_a.shape
    assert grads.lora_b.shape == dora.lora_b.shape
    assert grads.magnitude.shape == dora.magnitude.shape
