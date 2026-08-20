

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map


class AdagradState(NamedTuple):
    step: jnp.ndarray
    g2: Any  # running sum (not average) of squared grads


class Adagrad(Optimizer):

    lr: float = None
    eps: float = 1e-8
    initial_accumulator: float = 0.0

    def init(self, params):
        g2 = _tree_map(lambda p: jnp.full_like(p, self.initial_accumulator), params)
        return AdagradState(step=jnp.zeros([], jnp.int32), g2=g2)

    def update(self, grads, state, params=None, step=None):
        g2 = _tree_map(lambda g2, g: g2 + jnp.square(g), state.g2, grads)
        updates = _tree_map(
            lambda g, g2: -self.lr * g / (jnp.sqrt(g2) + self.eps), grads, g2
        )
        return updates, AdagradState(step=state.step + 1, g2=g2)


__all__ = ["Adagrad", "AdagradState"]
