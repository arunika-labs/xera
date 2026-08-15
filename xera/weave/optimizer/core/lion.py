"""Lion (Evolved Sign Momentum)."""

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map


class LionState(NamedTuple):
    step: jnp.ndarray
    m: Any


class Lion(Optimizer):

    def __init__(self, lr, b1=0.9, b2=0.99, weight_decay=0.0):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay

    def init(self, params):
        m = _tree_map(jnp.zeros_like, params)
        return LionState(step=jnp.zeros([], jnp.int32), m=m)

    def update(self, grads, state, params=None, step=None):
        direction = _tree_map(
            lambda m, g: jnp.sign(self.b1 * m + (1 - self.b1) * g),
            state.m, grads,
        )
        new_m = _tree_map(
            lambda m, g: self.b2 * m + (1 - self.b2) * g, state.m, grads
        )

        updates = _tree_map(lambda d: -self.lr * d, direction)

        if self.weight_decay and params is not None:
            updates = _tree_map(
                lambda u, p: u - self.lr * self.weight_decay * p, updates, params
            )

        return updates, LionState(step=state.step + 1, m=new_m)


__all__ = ["Lion", "LionState"]
