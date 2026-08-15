

from __future__ import annotations
import jax
import jax.numpy as jnp


def lecun_normal():
    
    def init(key, shape, dtype=jnp.float32):
        fan_in = shape[0]
        std = (1.0 / fan_in) ** 0.5
        return jax.random.normal(key, shape, dtype) * std
    return init


def zeros():
    
    def init(key, shape, dtype=jnp.float32):
        return jnp.zeros(shape, dtype)
    return init


def ones():
    
    def init(key, shape, dtype=jnp.float32):
        return jnp.ones(shape, dtype)
    return init


def uniform(scale=0.05):
    
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)
    return init


def normal(stddev=0.05):
    
    def init(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * stddev
    return init


def xavier_normal():
    
    def init(key, shape, dtype=jnp.float32):
        fan_in, fan_out = shape[0], shape[-1]
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        return jax.random.normal(key, shape, dtype) * std
    return init


def xavier_uniform():
    
    def init(key, shape, dtype=jnp.float32):
        fan_in, fan_out = shape[0], shape[-1]
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
    return init


def kaiming_normal():
    
    def init(key, shape, dtype=jnp.float32):
        fan_in = shape[0]
        std = (2.0 / fan_in) ** 0.5
        return jax.random.normal(key, shape, dtype) * std
    return init


def kaiming_uniform():
    
    def init(key, shape, dtype=jnp.float32):
        fan_in = shape[0]
        bound = (6.0 / fan_in) ** 0.5
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
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
]