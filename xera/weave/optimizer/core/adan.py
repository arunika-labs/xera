

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map


class AdanState(NamedTuple):
    step: jnp.ndarray
    m: Any        # momentum of grad
    v: Any        # momentum of grad difference
    n: Any        # second moment of (grad + (1-beta2)*diff)
    prev_grad: Any


class Adan(Optimizer):

    lr: float = None
    b1: float = 0.98
    b2: float = 0.92
    b3: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.0

    def init(self, params):
        m = _tree_map(jnp.zeros_like, params)
        v = _tree_map(jnp.zeros_like, params)
        n = _tree_map(jnp.zeros_like, params)
        prev_grad = _tree_map(jnp.zeros_like, params)
        return AdanState(
            step=jnp.zeros([], jnp.int32), m=m, v=v, n=n, prev_grad=prev_grad
        )

    def update(self, grads, state, params=None, step=None):
        step_ = state.step + 1
        is_first = state.step == 0

        # No previous gradient on the very first call -- treat the diff as 0
        # rather than (g - 0), which would otherwise spike v/n on step 1.
        diff = _tree_map(
            lambda g, pg: jnp.where(is_first, jnp.zeros_like(g), g - pg),
            grads, state.prev_grad,
        )

        m = _tree_map(lambda m, g: (1 - self.b1) * m + self.b1 * g, state.m, grads)
        v = _tree_map(lambda v, d: (1 - self.b2) * v + self.b2 * d, state.v, diff)
        n = _tree_map(
            lambda n, g, d: (1 - self.b3) * n
            + self.b3 * jnp.square(g + (1 - self.b2) * d),
            state.n, grads, diff,
        )

        updates = _tree_map(
            lambda m, v, n: -self.lr * (m + (1 - self.b2) * v) / (jnp.sqrt(n) + self.eps),
            m, v, n,
        )

        if self.weight_decay and params is not None:
            updates = _tree_map(
                lambda u, p: u - self.lr * self.weight_decay * p, updates, params
            )

        return updates, AdanState(step=step_, m=m, v=v, n=n, prev_grad=grads)


__all__ = ["Adan", "AdanState"]
