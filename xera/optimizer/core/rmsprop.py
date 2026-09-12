

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map, _as_hyper


class RMSpropState(NamedTuple):
    step: jnp.ndarray
    lr: jnp.ndarray
    decay: jnp.ndarray
    eps: jnp.ndarray
    momentum: jnp.ndarray  # coefficient; only used if `momentum_buf` isn't None
    v: Any                 # moving average of squared grad
    mean_g: Any             # moving average of grad (only used if centered)
    momentum_buf: Any       # only used if momentum > 0 at construction


class RMSprop(Optimizer):

    lr: float = None
    decay: float = 0.9
    eps: float = 1e-8
    momentum: float = 0.0
    # Whether the momentum/centered *buffers exist at all* is structural --
    # it's decided once, here, from the constructor value (self.momentum,
    # self.centered), and can't be changed mid-training without changing
    # the state's pytree shape. The momentum *coefficient* used once the
    # buffer exists is a separate, ordinary dynamic leaf (state.momentum)
    # and can still be scheduled freely.
    centered: bool = False

    def init(self, params):
        v = _tree_map(jnp.zeros_like, params)
        mean_g = _tree_map(jnp.zeros_like, params) if self.centered else None
        momentum_buf = _tree_map(jnp.zeros_like, params) if self.momentum else None
        return RMSpropState(
            step=jnp.zeros([], jnp.int32),
            lr=_as_hyper(self.lr), decay=_as_hyper(self.decay), eps=_as_hyper(self.eps),
            momentum=_as_hyper(self.momentum),
            v=v, mean_g=mean_g, momentum_buf=momentum_buf,
        )

    def update(self, grads, state, params=None, step=None):
        lr, decay, eps, momentum = state.lr, state.decay, state.eps, state.momentum

        v = _tree_map(
            lambda v, g: decay * v + (1 - decay) * jnp.square(g), state.v, grads,
        )

        if self.centered:
            mean_g = _tree_map(
                lambda mg, g: decay * mg + (1 - decay) * g, state.mean_g, grads,
            )
            denom = _tree_map(
                lambda v, mg: jnp.sqrt(v - jnp.square(mg)) + eps, v, mean_g
            )
        else:
            mean_g = None
            denom = _tree_map(lambda v: jnp.sqrt(v) + eps, v)

        direction = _tree_map(lambda g, d: g / d, grads, denom)

        if self.momentum:
            momentum_buf = _tree_map(
                lambda b, d: momentum * b + d, state.momentum_buf, direction
            )
            updates = _tree_map(lambda b: -lr * b, momentum_buf)
        else:
            momentum_buf = None
            updates = _tree_map(lambda d: -lr * d, direction)

        return updates, RMSpropState(
            step=state.step + 1, lr=lr, decay=decay, eps=eps, momentum=momentum,
            v=v, mean_g=mean_g, momentum_buf=momentum_buf,
        )


__all__ = ["RMSprop", "RMSpropState"]
