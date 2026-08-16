"""Adan (Adaptive Nesterov Momentum), Xie et al. 2022:
https://arxiv.org/abs/2208.06677

Tracks three moments: a momentum of the gradient, a momentum of the
gradient *difference* between consecutive steps, and a second-moment
estimate of a Nesterov-corrected gradient. This is a good-faith
implementation of the update rule from the paper, with weight decay
adapted to this library's `apply_updates(params, updates) = params +
updates` convention (an additive `-lr * weight_decay * params` term)
rather than the paper's multiplicative `params / (1 + lr * weight_decay)`
form -- the two are close for small `lr * weight_decay` but not identical.
This hasn't been numerically cross-checked against the authors' reference
implementation; treat it as a solid starting point rather than a verified
drop-in if you need to reproduce published numbers exactly.
"""

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

    def __init__(self, lr, b1=0.98, b2=0.92, b3=0.99, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.eps = eps
        self.weight_decay = weight_decay

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
