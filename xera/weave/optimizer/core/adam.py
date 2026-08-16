"""Adam and AdamW -- house every Adam-family variant here rather than
spinning up a new file per variant.
"""

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map


class AdamState(NamedTuple):
    step: jnp.ndarray
    m: Any
    v: Any


class Adam(Optimizer):
    """Plain Adam -- no weight decay at all. For decay, either use `AdamW`
    below (couples decay into its own update) or wrap this with
    `wrapper.WeightDecay` for a fully decoupled decay applied outside
    Adam's own moment estimates.
    """

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def init(self, params):
        m = _tree_map(jnp.zeros_like, params)
        v = _tree_map(jnp.zeros_like, params)
        return AdamState(step=jnp.zeros([], jnp.int32), m=m, v=v)

    def update(self, grads, state, params=None, step=None):
        step_ = state.step + 1
        step_f = step_.astype(jnp.float32)

        m = _tree_map(lambda m, g: self.b1 * m + (1 - self.b1) * g, state.m, grads)
        v = _tree_map(
            lambda v, g: self.b2 * v + (1 - self.b2) * jnp.square(g), state.v, grads
        )

        bias_c1 = 1 - self.b1 ** step_f
        bias_c2 = 1 - self.b2 ** step_f

        updates = _tree_map(
            lambda m, v: -self.lr * (m / bias_c1) / (jnp.sqrt(v / bias_c2) + self.eps),
            m, v,
        )

        return updates, AdamState(step=step_, m=m, v=v)


class AdamWState(NamedTuple):
    step: jnp.ndarray
    m: Any
    v: Any


class AdamW(Optimizer):

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay

    def init(self, params):
        m = _tree_map(jnp.zeros_like, params)
        v = _tree_map(jnp.zeros_like, params)
        return AdamWState(step=jnp.zeros([], jnp.int32), m=m, v=v)

    def update(self, grads, state, params=None, step=None):
        step_ = state.step + 1
        step_f = step_.astype(jnp.float32)

        m = _tree_map(lambda m, g: self.b1 * m + (1 - self.b1) * g, state.m, grads)
        v = _tree_map(
            lambda v, g: self.b2 * v + (1 - self.b2) * jnp.square(g), state.v, grads
        )

        bias_c1 = 1 - self.b1 ** step_f
        bias_c2 = 1 - self.b2 ** step_f

        updates = _tree_map(
            lambda m, v: -self.lr * (m / bias_c1) / (jnp.sqrt(v / bias_c2) + self.eps),
            m, v,
        )

        # Decoupled weight decay: applied directly to params, not folded into grads.
        if self.weight_decay and params is not None:
            updates = _tree_map(
                lambda u, p: u - self.lr * self.weight_decay * p, updates, params
            )

        return updates, AdamWState(step=step_, m=m, v=v)


__all__ = ["Adam", "AdamState", "AdamW", "AdamWState"]
