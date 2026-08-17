"""Tests for xera.loom.conv: Conv, ConvTranspose."""

import jax
import jax.numpy as jnp
import xera.loom as loom


def test_conv_forward_shape_and_grad():
    conv = loom.Conv(in_channels=3, out_channels=8, kernel_size=(3, 3), key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 3))
    out = conv(x)
    assert out.shape == (2, 16, 16, 8)  # SAME padding, stride 1

    grads = jax.grad(lambda c, x: jnp.sum(c(x) ** 2))(conv, x)
    assert grads.weight.shape == conv.weight.shape


def test_conv_grouped_depthwise():
    conv = loom.Conv(in_channels=4, out_channels=4, kernel_size=(3,), groups=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 4))
    out = conv(x)
    assert out.shape == (2, 10, 4)


def test_conv_stride_and_valid_padding_shrinks_output():
    conv = loom.Conv(
        in_channels=3, out_channels=4, kernel_size=(3, 3),
        stride=2, padding="VALID", key=jax.random.PRNGKey(0),
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 3))
    out = conv(x)
    # VALID padding + stride 2: (16 - 3) // 2 + 1 = 7
    assert out.shape == (2, 7, 7, 4)


def test_conv_transpose_upsamples():
    up = loom.ConvTranspose(in_channels=3, out_channels=8, kernel_size=(4, 4), stride=2, padding="SAME", key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 8, 3))
    out = up(x)
    assert out.shape == (2, 16, 16, 8)  # stride 2 doubles spatial size

    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(up, x)
    assert grads.weight.shape == up.weight.shape


def test_conv_then_conv_transpose_roundtrip_shape():
    down = loom.Conv(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=2, padding="SAME", key=jax.random.PRNGKey(0))
    up = loom.ConvTranspose(in_channels=8, out_channels=3, kernel_size=(3, 3), stride=2, padding="SAME", key=jax.random.PRNGKey(1))
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 16, 16, 3))
    assert up(down(x)).shape == x.shape
