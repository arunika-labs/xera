"""RMSprop, with optional momentum and the centered variant."""

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map


class RMSpropState(NamedTuple):
    step: jnp.ndarray
    v: Any            # moving average of squared grad
    mean_g: Any        # moving average of grad (only used if centered)
    momentum_buf: Any  # only used if momentum > 0


class RMSprop(Optimizer):
    """Classic RMSprop: divide the gradient by a running RMS of its recent
    magnitude.

    Args:
        decay: decay rate for the squared-grad running average (often
            called `alpha` or `rho` elsewhere).
        momentum: if > 0, applies plain momentum to the RMS-scaled
            gradient (à la PyTorch's `RMSprop(momentum=...)`), not
            Nesterov -- for Nesterov-style momentum, compose with a
            different core optimizer instead.
        centered: if True, also tracks a running mean of the gradient and
            uses `v - mean_g**2` as the normalizer (reduces bias from
            grads with a large mean vs. variance) instead of raw `v`.
    """

    def __init__(self, lr, decay=0.9, eps=1e-8, momentum=0.0, centered=False):
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.momentum = momentum
        self.centered = centered

    def init(self, params):
        v = _tree_map(jnp.zeros_like, params)
        mean_g = _tree_map(jnp.zeros_like, params) if self.centered else None
        momentum_buf = _tree_map(jnp.zeros_like, params) if self.momentum else None
        return RMSpropState(
            step=jnp.zeros([], jnp.int32), v=v, mean_g=mean_g, momentum_buf=momentum_buf
        )

    def update(self, grads, state, params=None, step=None):
        v = _tree_map(
            lambda v, g: self.decay * v + (1 - self.decay) * jnp.square(g),
            state.v, grads,
        )

        if self.centered:
            mean_g = _tree_map(
                lambda mg, g: self.decay * mg + (1 - self.decay) * g,
                state.mean_g, grads,
            )
            denom = _tree_map(
                lambda v, mg: jnp.sqrt(v - jnp.square(mg)) + self.eps, v, mean_g
            )
        else:
            mean_g = None
            denom = _tree_map(lambda v: jnp.sqrt(v) + self.eps, v)

        direction = _tree_map(lambda g, d: g / d, grads, denom)

        if self.momentum:
            momentum_buf = _tree_map(
                lambda b, d: self.momentum * b + d, state.momentum_buf, direction
            )
            updates = _tree_map(lambda b: -self.lr * b, momentum_buf)
        else:
            momentum_buf = None
            updates = _tree_map(lambda d: -self.lr * d, direction)

        return updates, RMSpropState(
            step=state.step + 1, v=v, mean_g=mean_g, momentum_buf=momentum_buf
        )


__all__ = ["RMSprop", "RMSpropState"]
