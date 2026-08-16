from __future__ import annotations
import jax
import jax.numpy as jnp


def _fan_in_out(shape):
    """Compute (fan_in, fan_out) for a param shape.

    - 2D `(in, out)` (Dense-style): fan_in = shape[0], fan_out = shape[1].
    - >2D `(*kernel_size, in_channels, out_channels)` (Conv-style): fan_in
      and fan_out both need to account for the receptive field, not just
      the raw channel counts -- e.g. a (3, 3, 16, 32) conv kernel has
      fan_in = 16 * 3 * 3, not 3 (which `shape[0]` would incorrectly give).
    - 1D or 0D: no real in/out distinction: fall back to shape[0] for both
      (matches prior behavior for these degenerate cases, e.g. bias vectors
      that get zero-init anyway and never actually call this).
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
    
    def init(key, shape, dtype=jnp.float32):
        fan_in, _ = _fan_in_out(shape)
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
        fan_in, fan_out = _fan_in_out(shape)
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        return jax.random.normal(key, shape, dtype) * std
    return init


def xavier_uniform():
    
    def init(key, shape, dtype=jnp.float32):
        fan_in, fan_out = _fan_in_out(shape)
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
    return init


def kaiming_normal():
    
    def init(key, shape, dtype=jnp.float32):
        fan_in, _ = _fan_in_out(shape)
        std = (2.0 / fan_in) ** 0.5
        return jax.random.normal(key, shape, dtype) * std
    return init


def kaiming_uniform():
    
    def init(key, shape, dtype=jnp.float32):
        fan_in, _ = _fan_in_out(shape)
        bound = (6.0 / fan_in) ** 0.5
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
    return init


def constant(value=0.0):
    
    def init(key, shape, dtype=jnp.float32):
        return jnp.full(shape, value, dtype)
    return init


def truncated_normal(stddev=0.05):
    
    def init(key, shape, dtype=jnp.float32):
        return jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype) * stddev
    return init


def orthogonal(scale=1.0):
    

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
