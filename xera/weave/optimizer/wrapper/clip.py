

from __future__ import annotations
import jax.numpy as jnp
from ..base import Optimizer, _tree_map, _global_norm
from ...struct import Struct


class Clip(Struct):

    threshold: float = None

    def setup(self):
        self.threshold = float(self.threshold)

    def __call__(self, inner: Optimizer) -> Optimizer:
        return _Clipped(inner, self.threshold)


class _Clipped(Optimizer):
    inner: Optimizer = None
    threshold: float = None

    def init(self, params):
        return self.inner.init(params)

    def update(self, grads, state, params=None, step=None):
        norm = _global_norm(grads)
        scale = jnp.minimum(1.0, self.threshold / (norm + 1e-7))
        clipped = _tree_map(lambda g: g * scale, grads)
        return self.inner.update(clipped, state, params, step)


__all__ = ["Clip"]
