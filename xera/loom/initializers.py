"""
Parameter initialization functions for neural network layers.

This module provides a comprehensive collection of weight initialization
schemes commonly used in deep learning, including LeCun, Xavier, Kaiming,
and orthogonal initializations.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp


def _fan_in_out(shape):
    """
    Compute fan-in and fan-out for a given weight shape.
    
    Fan-in is the number of input units to a weight tensor, and fan-out
    is the number of output units. These are used to scale initializations
    appropriately for the layer size.
    
    Args:
        shape: The shape of the weight tensor.
    
    Returns:
        A tuple (fan_in, fan_out).
    """
    if len(shape) == 2:
        return shape[0], shape[1]
    elif len(shape) > 2:
        receptive_field_size = 1
        for s in shape[:-2]:
            receptive_field_size *= s
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
        return fan_in, fan_out
    else:
        fan = shape[0] if shape else 1
        return fan, fan


def lecun_normal():
    """
    LeCun normal initialization.
    
    Initializes weights with a normal distribution scaled by fan_in.
    Suitable for layers with SELU or linear activations.
    
    Returns:
        An initialization function that takes (key, shape, dtype) and returns
        initialized weights.
    """
    def init(key, shape, dtype=jnp.float32):
        fan_in, _ = _fan_in_out(shape)
        std = (1.0 / fan_in) ** 0.5
        return jax.random.normal(key, shape, dtype) * std
    return init


def zeros():
    """
    Initialize weights to zeros.
    
    Returns:
        An initialization function that takes (key, shape, dtype) and returns
        a zero array.
    """
    def init(key, shape, dtype=jnp.float32):
        return jnp.zeros(shape, dtype)
    return init


def ones():
    """
    Initialize weights to ones.
    
    Returns:
        An initialization function that takes (key, shape, dtype) and returns
        a ones array.
    """
    def init(key, shape, dtype=jnp.float32):
        return jnp.ones(shape, dtype)
    return init


def uniform(scale=0.05):
    """
    Uniform initialization.
    
    Initializes weights with a uniform distribution in [-scale, scale].
    
    Args:
        scale: The range of the uniform distribution (default: 0.05).
    
    Returns:
        An initialization function that takes (key, shape, dtype) and returns
        initialized weights.
    """
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)
    return init


def normal(stddev=0.05):
    """
    Normal (Gaussian) initialization.
    
    Initializes weights with a normal distribution with given standard deviation.
    
    Args:
        stddev: Standard deviation of the normal distribution (default: 0.05).
    
    Returns:
        An initialization function that takes (key, shape, dtype) and returns
        initialized weights.
    """
    def init(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * stddev
    return init


def xavier_normal():
    """
    Xavier/Glorot normal initialization.
    
    Initializes weights with a normal distribution scaled by the average
    of fan_in and fan_out. Suitable for layers with tanh or sigmoid activations.
    
    Returns:
        An initialization function that takes (key, shape, dtype) and returns
        initialized weights.
    """
    def init(key, shape, dtype=jnp.float32):
        fan_in, fan_out = _fan_in_out(shape)
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        return jax.random.normal(key, shape, dtype) * std
    return init


def xavier_uniform():
    """
    Xavier/Glorot uniform initialization.
    
    Initializes weights with a uniform distribution scaled by the average
    of fan_in and fan_out. Suitable for layers with tanh or sigmoid activations.
    
    Returns:
        An initialization function that takes (key, shape, dtype) and returns
        initialized weights.
    """
    def init(key, shape, dtype=jnp.float32):
        fan_in, fan_out = _fan_in_out(shape)
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
    return init


def kaiming_normal():
    """
    Kaiming/He normal initialization.
    
    Initializes weights with a normal distribution scaled by fan_in.
    Suitable for layers with ReLU or leaky ReLU activations.
    
    Returns:
        An initialization function that takes (key, shape, dtype) and returns
        initialized weights.
    """
    def init(key, shape, dtype=jnp.float32):
        fan_in, _ = _fan_in_out(shape)
        std = (2.0 / fan_in) ** 0.5
        return jax.random.normal(key, shape, dtype) * std
    return init


def kaiming_uniform():
    """
    Kaiming/He uniform initialization.
    
    Initializes weights with a uniform distribution scaled by fan_in.
    Suitable for layers with ReLU or leaky ReLU activations.
    
    Returns:
        An initialization function that takes (key, shape, dtype) and returns
        initialized weights.
    """
    def init(key, shape, dtype=jnp.float32):
        fan_in, _ = _fan_in_out(shape)
        bound = (6.0 / fan_in) ** 0.5
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
    return init


def constant(value=0.0):
    """
    Constant initialization.
    
    Initializes weights to a constant value.
    
    Args:
        value: The constant value to initialize with (default: 0.0).
    
    Returns:
        An initialization function that takes (key, shape, dtype) and returns
        initialized weights.
    """
    def init(key, shape, dtype=jnp.float32):
        return jnp.full(shape, value, dtype)
    return init


def truncated_normal(stddev=0.05):
    """
    Truncated normal initialization.
    
    Initializes weights with a truncated normal distribution (values more
    than 2 standard deviations from the mean are discarded and re-drawn).
    
    Args:
        stddev: Standard deviation of the normal distribution (default: 0.05).
    
    Returns:
        An initialization function that takes (key, shape, dtype) and returns
        initialized weights.
    """
    def init(key, shape, dtype=jnp.float32):
        return jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype) * stddev
    return init


def orthogonal(scale=1.0):
    """
    Orthogonal initialization.
    
    Initializes weights with an orthogonal matrix (or a semi-orthogonal matrix
    for non-square shapes). This helps preserve gradient flow in deep networks.
    
    Args:
        scale: Scaling factor for the orthogonal matrix (default: 1.0).
    
    Returns:
        An initialization function that takes (key, shape, dtype) and returns
        initialized weights.
    
    Raises:
        ValueError: If the shape has fewer than 2 dimensions.
    """
    def init(key, shape, dtype=jnp.float32):
        if len(shape) < 2:
            raise ValueError("orthogonal init requires a shape with at least 2 dimensions.")
        n_rows = shape[0]
        n_cols = int(jnp.prod(jnp.array(shape[1:])))
        flat_shape = (max(n_rows, n_cols), min(n_rows, n_cols))
        a = jax.random.normal(key, flat_shape, dtype)
        q, r = jnp.linalg.qr(a)
        
        d = jnp.sign(jnp.diagonal(r))
        q = q * d
        if n_rows < n_cols:
            q = q.T
        q = q.reshape(shape)
        return scale * q
    return init


def variance_scaling(scale=1.0, mode="fan_in", distribution="normal"):
    """
    Variance scaling initialization.
    
    A flexible initialization scheme that can reproduce many common initializers
    by adjusting scale, mode, and distribution parameters.
    
    Args:
        scale: Scaling factor for the variance.
        mode: How to compute the fan: "fan_in", "fan_out", or "fan_avg".
        distribution: Distribution to use: "normal", "truncated_normal", or "uniform".
    
    Returns:
        An initialization function that takes (key, shape, dtype) and returns
        initialized weights.
    
    Raises:
        ValueError: If mode or distribution is not recognized.
    """
    def init(key, shape, dtype=jnp.float32):
        fan_in, fan_out = _fan_in_out(shape)
        if mode == "fan_in":
            denom = fan_in
        elif mode == "fan_out":
            denom = fan_out
        elif mode == "fan_avg":
            denom = (fan_in + fan_out) / 2.0
        else:
            raise ValueError(f"unknown mode: {mode!r}")

        variance = scale / denom
        if distribution == "normal":
            return jax.random.normal(key, shape, dtype) * (variance ** 0.5)
        elif distribution == "truncated_normal":
            # Adjust for truncation to maintain variance
            stddev = (variance ** 0.5) / 0.87962566103423978
            return jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype) * stddev
        elif distribution == "uniform":
            bound = (3.0 * variance) ** 0.5
            return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
        else:
            raise ValueError(f"unknown distribution: {distribution!r}")
    return init


__all__ = [
    "lecun_normal",
    "zeros",
    "ones",
    "uniform",
    "normal",
    "xavier_normal",
    "xavier_uniform",
    "kaiming_normal",
    "kaiming_uniform",
    "constant",
    "truncated_normal",
    "orthogonal",
    "variance_scaling",
]
