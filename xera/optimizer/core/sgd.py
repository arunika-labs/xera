

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map, _as_hyper


class SGDMomentumState(NamedTuple):
    step: jnp.ndarray
    lr: jnp.ndarray
    momentum: jnp.ndarray
    weight_decay: jnp.ndarray
    momentum_buf: Any


class SGDMomentum(Optimizer):

    lr: float = None
    momentum: float = 0.9
    nesterov: bool = False  # structural: gates a Python `if`, stays static
    weight_decay: float = 0.0

    def init(self, params):
        m = _tree_map(jnp.zeros_like, params)
        return SGDMomentumState(
            step=jnp.zeros([], jnp.int32),
            lr=_as_hyper(self.lr), momentum=_as_hyper(self.momentum),
            weight_decay=_as_hyper(self.weight_decay),
            momentum_buf=m,
        )

    def update(self, grads, state, params=None, step=None):
        lr, momentum, weight_decay = state.lr, state.momentum, state.weight_decay

        if params is not None:
            grads = _tree_map(lambda g, p: g + weight_decay * p, grads, params)

        new_m = _tree_map(
            lambda m, g: momentum * m + g, state.momentum_buf, grads
        )

        if self.nesterov:
            direction = _tree_map(lambda m, g: momentum * m + g, new_m, grads)
        else:
            direction = new_m

        updates = _tree_map(lambda d: -lr * d, direction)
        return updates, SGDMomentumState(
            step=state.step + 1, lr=lr, momentum=momentum,
            weight_decay=weight_decay, momentum_buf=new_m,
        )


__all__ = ["SGDMomentum", "SGDMomentumState"]
