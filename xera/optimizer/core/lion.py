

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map, _as_hyper


class LionState(NamedTuple):
    step: jnp.ndarray
    lr: jnp.ndarray
    b1: jnp.ndarray
    b2: jnp.ndarray
    weight_decay: jnp.ndarray
    m: Any


class Lion(Optimizer):

    lr: float = None
    b1: float = 0.9
    b2: float = 0.99
    weight_decay: float = 0.0

    def init(self, params):
        m = _tree_map(jnp.zeros_like, params)
        return LionState(
            step=jnp.zeros([], jnp.int32),
            lr=_as_hyper(self.lr), b1=_as_hyper(self.b1), b2=_as_hyper(self.b2),
            weight_decay=_as_hyper(self.weight_decay),
            m=m,
        )

    def update(self, grads, state, params=None, step=None):
        lr, b1, b2, weight_decay = state.lr, state.b1, state.b2, state.weight_decay

        direction = _tree_map(
            lambda m, g: jnp.sign(b1 * m + (1 - b1) * g),
            state.m, grads,
        )
        new_m = _tree_map(lambda m, g: b2 * m + (1 - b2) * g, state.m, grads)

        updates = _tree_map(lambda d: -lr * d, direction)

        if params is not None:
            updates = _tree_map(
                lambda u, p: u - lr * weight_decay * p, updates, params
            )

        return updates, LionState(
            step=state.step + 1, lr=lr, b1=b1, b2=b2,
            weight_decay=weight_decay, m=new_m,
        )


__all__ = ["Lion", "LionState"]
