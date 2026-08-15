"""Clip: global-norm gradient clipping wrapper. Generic -- wraps any
Optimizer, doesn't know or care what it is.
"""

from __future__ import annotations
import jax.numpy as jnp
from ..base import Optimizer, _tree_map, _global_norm


class Clip:
    """Factory: clips gradients by global norm before the wrapped optimizer
    sees them (standard global-norm gradient clipping, à la
    optax.clip_by_global_norm). This clips the *raw gradient tree*, which is
    a different thing from Muon's own per-leaf direction clipping inside
    MuonCore (that one clips the momentum direction of a single leaf right
    before orthogonalizing it, not the whole tree's gradient norm) -- the
    two are unrelated and can be used together.

    Usage:
        opt = O.Clip(1.0)(O.AdamW(lr=1e-4))
    """

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def __call__(self, inner: Optimizer) -> Optimizer:
        return _Clipped(inner, self.threshold)


class _Clipped(Optimizer):
    def __init__(self, inner, threshold):
        self.inner = inner
        self.threshold = threshold

    def init(self, params):
        return self.inner.init(params)

    def update(self, grads, state, params=None, step=None):
        norm = _global_norm(grads)
        scale = jnp.minimum(1.0, self.threshold / (norm + 1e-7))
        clipped = _tree_map(lambda g: g * scale, grads)
        return self.inner.update(clipped, state, params, step)


__all__ = ["Clip"]
