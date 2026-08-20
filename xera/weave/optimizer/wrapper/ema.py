

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map
from ...struct import Struct


class EMA(Struct):

    decay: float = 0.999
    warmup_steps: int = 0

    def setup(self):
        self.decay = float(self.decay)
        self.warmup_steps = int(self.warmup_steps)

    def __call__(self, inner: Optimizer) -> Optimizer:
        return _EMAed(inner, self.decay, self.warmup_steps)


class _EMAedState(NamedTuple):
    inner_state: Any
    shadow: Any
    step: jnp.ndarray


class _EMAed(Optimizer):
    inner: Optimizer = None
    decay: float = None
    warmup_steps: int = None

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
        return state.shadow


__all__ = ["EMA"]
