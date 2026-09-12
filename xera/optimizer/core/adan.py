

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map, _as_hyper


class AdanState(NamedTuple):
    step: jnp.ndarray
    lr: jnp.ndarray
    b1: jnp.ndarray
    b2: jnp.ndarray
    b3: jnp.ndarray
    eps: jnp.ndarray
    weight_decay: jnp.ndarray
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
            step=jnp.zeros([], jnp.int32),
            lr=_as_hyper(self.lr), b1=_as_hyper(self.b1), b2=_as_hyper(self.b2),
            b3=_as_hyper(self.b3), eps=_as_hyper(self.eps),
            weight_decay=_as_hyper(self.weight_decay),
            m=m, v=v, n=n, prev_grad=prev_grad,
        )

    def update(self, grads, state, params=None, step=None):
        step_ = state.step + 1
        is_first = state.step == 0
        lr, b1, b2, b3 = state.lr, state.b1, state.b2, state.b3
        eps, weight_decay = state.eps, state.weight_decay

        # No previous gradient on the very first call -- treat the diff as 0
        # rather than (g - 0), which would otherwise spike v/n on step 1.
        diff = _tree_map(
            lambda g, pg: jnp.where(is_first, jnp.zeros_like(g), g - pg),
            grads, state.prev_grad,
        )

        m = _tree_map(lambda m, g: (1 - b1) * m + b1 * g, state.m, grads)
        v = _tree_map(lambda v, d: (1 - b2) * v + b2 * d, state.v, diff)
        n = _tree_map(
            lambda n, g, d: (1 - b3) * n + b3 * jnp.square(g + (1 - b2) * d),
            state.n, grads, diff,
        )

        updates = _tree_map(
            lambda m, v, n: -lr * (m + (1 - b2) * v) / (jnp.sqrt(n) + eps),
            m, v, n,
        )

        if params is not None:
            updates = _tree_map(
                lambda u, p: u - lr * weight_decay * p, updates, params
            )

        return updates, AdanState(
            step=step_, lr=lr, b1=b1, b2=b2, b3=b3, eps=eps,
            weight_decay=weight_decay, m=m, v=v, n=n, prev_grad=grads,
        )


__all__ = ["Adan", "AdanState"]
