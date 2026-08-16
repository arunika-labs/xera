"""EMA: exponential moving average of *params* (not gradients -- that's
Accumulate's job). Common for eval/checkpointing: train with the raw
weights but evaluate with their smoothed shadow copy.
"""

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map


class EMA:
    """Factory: maintains a decayed shadow copy of params alongside the
    wrapped optimizer's normal training. Doesn't change what gets trained
    -- `updates` returned are exactly the wrapped optimizer's own updates,
    unmodified -- it just tracks a smoothed copy on the side.

    Usage:
        opt = O.EMA(0.999)(O.AdamW(lr=1e-3))
        state = opt.init(params)
        ...
        updates, state = opt.update(grads, state, params)
        params = O.apply_updates(params, updates)
        eval_params = opt.ema_params(state)   # smoothed weights for eval

    `warmup_steps`: before this many steps, the shadow is set to the raw
    params outright (decay=0) rather than partially mixed -- avoids the
    shadow being dominated by the (usually poor) initialization for a long
    time when `decay` is close to 1.

    Step source: uses the external `step` if threaded via
    `update(..., step=...)`, otherwise an internal counter -- same
    convention as Schedule/Accumulate.
    """

    def __init__(self, decay: float = 0.999, warmup_steps: int = 0):
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)

    def __call__(self, inner: Optimizer) -> Optimizer:
        return _EMAed(inner, self.decay, self.warmup_steps)


class _EMAedState(NamedTuple):
    inner_state: Any
    shadow: Any
    step: jnp.ndarray


class _EMAed(Optimizer):
    def __init__(self, inner, decay, warmup_steps):
        self.inner = inner
        self.decay = decay
        self.warmup_steps = warmup_steps

    def init(self, params):
        return _EMAedState(
            inner_state=self.inner.init(params),
            shadow=params,
            step=jnp.zeros([], jnp.int32),
        )

    def update(self, grads, state, params=None, step=None):
        updates, new_inner = self.inner.update(grads, state.inner_state, params, step)

        use_step = state.step if step is None else jnp.asarray(step)
        decay = jnp.where(use_step < self.warmup_steps, 0.0, self.decay)

        if params is not None:
            new_params = _tree_map(lambda p, u: p + u, params, updates)
            shadow = _tree_map(
                lambda s, p: decay * s + (1 - decay) * p, state.shadow, new_params
            )
        else:
            shadow = state.shadow

        return updates, _EMAedState(
            inner_state=new_inner, shadow=shadow, step=state.step + 1
        )

    def ema_params(self, state):
        """Return the EMA-smoothed shadow params (e.g. for evaluation or
        checkpointing) -- not part of the base Optimizer interface, since
        no other wrapper needs a second, non-`updates` output like this.
        """
        return state.shadow


__all__ = ["EMA"]
