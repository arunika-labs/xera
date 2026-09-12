

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map, _as_hyper


class AdagradState(NamedTuple):
    step: jnp.ndarray
    lr: jnp.ndarray
    eps: jnp.ndarray
    g2: Any  # running sum (not average) of squared grads


class Adagrad(Optimizer):

    lr: float = None
    eps: float = 1e-8
    # Only used once, to seed g2 at init -- not read again in update(), so
    # there's no per-step arithmetic for it to drive. Left as static config
    # rather than a state leaf for that reason.
    initial_accumulator: float = 0.0

    def init(self, params):
        g2 = _tree_map(lambda p: jnp.full_like(p, self.initial_accumulator), params)
        return AdagradState(
            step=jnp.zeros([], jnp.int32),
            lr=_as_hyper(self.lr), eps=_as_hyper(self.eps),
            g2=g2,
        )

    def update(self, grads, state, params=None, step=None):
        lr, eps = state.lr, state.eps
        g2 = _tree_map(lambda g2, g: g2 + jnp.square(g), state.g2, grads)
        updates = _tree_map(
            lambda g, g2: -lr * g / (jnp.sqrt(g2) + eps), grads, g2
        )
        return updates, AdagradState(step=state.step + 1, lr=lr, eps=eps, g2=g2)


__all__ = ["Adagrad", "AdagradState"]
