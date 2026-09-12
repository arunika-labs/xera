

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map, _global_norm, _as_hyper
from ..state import State


class Clip(State):

    threshold: float = None

    def __call__(self, inner: Optimizer) -> Optimizer:
        return _Clipped(inner, self.threshold)


class _ClippedState(NamedTuple):
    inner_state: Any
    threshold: jnp.ndarray


class _Clipped(Optimizer):
    inner: Optimizer = None
    threshold: float = None

    def init(self, params):
        return _ClippedState(
            inner_state=self.inner.init(params), threshold=_as_hyper(self.threshold)
        )

    def update(self, grads, state, params=None, step=None):
        threshold = state.threshold
        norm = _global_norm(grads)
        scale = jnp.minimum(1.0, threshold / (norm + 1e-7))
        clipped = _tree_map(lambda g: g * scale, grads)
        updates, new_inner = self.inner.update(clipped, state.inner_state, params, step)
        return updates, _ClippedState(inner_state=new_inner, threshold=threshold)


__all__ = ["Clip"]
