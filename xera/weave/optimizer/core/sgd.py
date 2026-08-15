"""SGD with (optionally Nesterov) momentum."""

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map


class SGDMomentumState(NamedTuple):
    step: jnp.ndarray
    momentum: Any


class SGDMomentum(Optimizer):

    def __init__(self, lr, momentum=0.9, nesterov=False, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay

    def init(self, params):
        m = _tree_map(jnp.zeros_like, params)
        return SGDMomentumState(step=jnp.zeros([], jnp.int32), momentum=m)

    def update(self, grads, state, params=None, step=None):
        if self.weight_decay and params is not None:
            grads = _tree_map(
                lambda g, p: g + self.weight_decay * p, grads, params
            )

        new_m = _tree_map(
            lambda m, g: self.momentum * m + g, state.momentum, grads
        )

        if self.nesterov:
            direction = _tree_map(
                lambda m, g: self.momentum * m + g, new_m, grads
            )
        else:
            direction = new_m

        updates = _tree_map(lambda d: -self.lr * d, direction)
        return updates, SGDMomentumState(step=state.step + 1, momentum=new_m)


__all__ = ["SGDMomentum", "SGDMomentumState"]
