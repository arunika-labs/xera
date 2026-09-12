

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map, _as_hyper


class AdamState(NamedTuple):
    step: jnp.ndarray
    lr: jnp.ndarray
    b1: jnp.ndarray
    b2: jnp.ndarray
    eps: jnp.ndarray
    m: Any
    v: Any


class Adam(Optimizer):

    lr: float = None
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8

    def init(self, params):
        m = _tree_map(jnp.zeros_like, params)
        v = _tree_map(jnp.zeros_like, params)
        return AdamState(
            step=jnp.zeros([], jnp.int32),
            lr=_as_hyper(self.lr), b1=_as_hyper(self.b1),
            b2=_as_hyper(self.b2), eps=_as_hyper(self.eps),
            m=m, v=v,
        )

    def update(self, grads, state, params=None, step=None):
        step_ = state.step + 1
        step_f = step_.astype(jnp.float32)
        lr, b1, b2, eps = state.lr, state.b1, state.b2, state.eps

        m = _tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.m, grads)
        v = _tree_map(lambda v, g: b2 * v + (1 - b2) * jnp.square(g), state.v, grads)

        bias_c1 = 1 - b1 ** step_f
        bias_c2 = 1 - b2 ** step_f

        updates = _tree_map(
            lambda m, v: -lr * (m / bias_c1) / (jnp.sqrt(v / bias_c2) + eps),
            m, v,
        )

        return updates, AdamState(
            step=step_, lr=lr, b1=b1, b2=b2, eps=eps, m=m, v=v
        )


class AdamWState(NamedTuple):
    step: jnp.ndarray
    lr: jnp.ndarray
    b1: jnp.ndarray
    b2: jnp.ndarray
    eps: jnp.ndarray
    weight_decay: jnp.ndarray
    m: Any
    v: Any


class AdamW(Optimizer):

    lr: float = None
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01

    def init(self, params):
        m = _tree_map(jnp.zeros_like, params)
        v = _tree_map(jnp.zeros_like, params)
        return AdamWState(
            step=jnp.zeros([], jnp.int32),
            lr=_as_hyper(self.lr), b1=_as_hyper(self.b1),
            b2=_as_hyper(self.b2), eps=_as_hyper(self.eps),
            weight_decay=_as_hyper(self.weight_decay),
            m=m, v=v,
        )

    def update(self, grads, state, params=None, step=None):
        step_ = state.step + 1
        step_f = step_.astype(jnp.float32)
        lr, b1, b2, eps = state.lr, state.b1, state.b2, state.eps
        weight_decay = state.weight_decay

        m = _tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.m, grads)
        v = _tree_map(lambda v, g: b2 * v + (1 - b2) * jnp.square(g), state.v, grads)

        bias_c1 = 1 - b1 ** step_f
        bias_c2 = 1 - b2 ** step_f

        updates = _tree_map(
            lambda m, v: -lr * (m / bias_c1) / (jnp.sqrt(v / bias_c2) + eps),
            m, v,
        )

        # Decoupled weight decay: applied directly to params, not folded into
        # grads. Applied unconditionally (a weight_decay=0 leaf is a no-op)
        # rather than gated by a Python `if`, so weight_decay stays a normal
        # dynamic/schedulable leaf instead of a structural switch.
        if params is not None:
            updates = _tree_map(
                lambda u, p: u - lr * weight_decay * p, updates, params
            )

        return updates, AdamWState(
            step=step_, lr=lr, b1=b1, b2=b2, eps=eps,
            weight_decay=weight_decay, m=m, v=v,
        )


__all__ = ["Adam", "AdamState", "AdamW", "AdamWState"]
