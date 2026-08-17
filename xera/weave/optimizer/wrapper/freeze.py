

from __future__ import annotations
from typing import NamedTuple
import jax.numpy as jnp
from ..base import Optimizer, _tree_map
from ..partition import Partition


class _NoOpState(NamedTuple):
    step: jnp.ndarray


class _NoOp(Optimizer):


    def init(self, params):
        return _NoOpState(step=jnp.zeros([], jnp.int32))

    def update(self, grads, state, params=None, step=None):
        zero = _tree_map(jnp.zeros_like, grads)
        return zero, _NoOpState(step=state.step + 1)


class Freeze:


    def __init__(self, predicate):
        self.predicate = predicate

    def __call__(self, inner: Optimizer) -> Optimizer:
        return Partition([
            (self.predicate, _NoOp()),
            (lambda path, leaf: True, inner),
        ])


__all__ = ["Freeze"]
