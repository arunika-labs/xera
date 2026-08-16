

from __future__ import annotations
import jax
import jax.numpy as jnp
from ..core import Module


def _as_tuple(v, n):
    return v if isinstance(v, tuple) else (v,) * n


class MaxPool(Module):
    """Max pooling over spatial dims, rank-agnostic like `Conv` (pool size
    tuple length determines how many spatial dims are pooled). Channel-last
    input: `(batch, *spatial, channels)`. No learnable params.
    """
    pool_size: tuple
    stride: tuple = None
    padding: str = "VALID"

    def __call__(self, x):
        n_spatial = len(self.pool_size)
        stride = self.stride if self.stride is not None else self.pool_size
        window_dims = (1,) + tuple(self.pool_size) + (1,)
        window_strides = (1,) + _as_tuple(stride, n_spatial) + (1,)
        return jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, window_dims, window_strides, self.padding
        )


class AvgPool(Module):
    """Average pooling over spatial dims, rank-agnostic like `Conv`.
    Channel-last input: `(batch, *spatial, channels)`. No learnable params.

    Correct under both `"VALID"` and `"SAME"` padding -- divides by the
    actual number of in-bounds elements per window (computed via reducing
    a same-shaped ones array with the same window/padding), not a fixed
    `prod(pool_size)`, which would be wrong for edge windows under `"SAME"`.
    """
    pool_size: tuple
    stride: tuple = None
    padding: str = "VALID"

    def __call__(self, x):
        n_spatial = len(self.pool_size)
        stride = self.stride if self.stride is not None else self.pool_size
        window_dims = (1,) + tuple(self.pool_size) + (1,)
        window_strides = (1,) + _as_tuple(stride, n_spatial) + (1,)

        summed = jax.lax.reduce_window(
            x, 0.0, jax.lax.add, window_dims, window_strides, self.padding
        )
        counts = jax.lax.reduce_window(
            jnp.ones_like(x), 0.0, jax.lax.add, window_dims, window_strides, self.padding
        )
        return summed / counts


class GlobalAvgPool(Module):
    """Averages every spatial dim away entirely -- `(batch, *spatial,
    channels)` -> `(batch, channels)` (or `(batch, *1s, channels)` if
    `keepdims=True`). The common "pool to a single vector per example"
    layer before a classifier head. No learnable params.
    """
    keepdims: bool = False

    def __call__(self, x):
        spatial_axes = tuple(range(1, x.ndim - 1))
        return jnp.mean(x, axis=spatial_axes, keepdims=self.keepdims)


__all__ = ["MaxPool", "AvgPool", "GlobalAvgPool"]
