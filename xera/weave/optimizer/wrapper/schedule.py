

from __future__ import annotations
from typing import NamedTuple, Any, Callable
import jax.numpy as jnp
from ..base import Optimizer, _tree_map
from ...struct import Struct


class Schedule(Struct):

    fn: Callable[[jnp.ndarray], jnp.ndarray] = None

    def __call__(self, inner: Optimizer) -> Optimizer:
        return _Scheduled(inner, self.fn)


class _ScheduledState(NamedTuple):
    inner_state: Any
    step: jnp.ndarray


class _Scheduled(Optimizer):
    inner: Optimizer = None
    fn: Callable = None

    def init(self, params):
        return _ScheduledState(
            inner_state=self.inner.init(params), step=jnp.zeros([], jnp.int32)
        )

    def update(self, grads, state, params=None, step=None):
        use_step = state.step if step is None else jnp.asarray(step)
        scale = self.fn(use_step)
        updates, new_inner = self.inner.update(grads, state.inner_state, params, step)
        updates = _tree_map(lambda u: u * scale, updates)
        return updates, _ScheduledState(inner_state=new_inner, step=state.step + 1)


__all__ = ["Schedule"]
