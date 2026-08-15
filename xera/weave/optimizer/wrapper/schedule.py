"""Schedule: scales a wrapped optimizer's updates by fn(step). Generic --
wraps any Optimizer, doesn't know or care what it is.
"""

from __future__ import annotations
from typing import NamedTuple, Any, Callable
import jax.numpy as jnp
from ..base import Optimizer, _tree_map


class Schedule:
    """Factory: scales the wrapped optimizer's *updates* by `fn(step)` on
    every call.

    `fn(step) -> scalar` should return a multiplier relative to the wrapped
    optimizer's own base lr (e.g. a cosine curve from 1.0 down to 0.0), NOT
    an absolute learning rate -- the wrapped optimizer already has its own
    `lr` baked in; Schedule only rescales its output.

    Step source and composition ordering:
    If an external `step` is threaded in via `update(..., step=...)`,
    Schedule calls `fn` with exactly that value. Otherwise it keeps its own
    internal counter that increments once per call. This matters when
    Schedule is composed with Accumulate:

        Schedule(cosine)(Accumulate(4)(Muon()))   # fn sees every micro-step
        Accumulate(4)(Schedule(cosine)(Muon()))   # fn only sees macro-steps
                                                    # (once per 4 micro-steps)

    Without an explicit `step`, these two have different lr curves even
    though the wrapper list is the same set of names -- purely because of
    nesting order. Threading an explicit `step` from the training loop
    removes this ambiguity: both wrappers read the same counter, and which
    one they see (raw micro-step vs. macro-step) becomes something you
    choose explicitly by what you pass as `step`, not an accident of
    nesting order.
    """

    def __init__(self, fn: Callable[[jnp.ndarray], jnp.ndarray]):
        self.fn = fn

    def __call__(self, inner: Optimizer) -> Optimizer:
        return _Scheduled(inner, self.fn)


class _ScheduledState(NamedTuple):
    inner_state: Any
    step: jnp.ndarray


class _Scheduled(Optimizer):
    def __init__(self, inner, fn):
        self.inner = inner
        self.fn = fn

    def init(self, params):
        return _ScheduledState(
            inner_state=self.inner.init(params), step=jnp.zeros([], jnp.int32)
        )

    def update(self, grads, state, params=None, step=None):
        use_step = state.step if step is None else jnp.asarray(step)
        scale = self.fn(use_step)
        updates, new_inner = self.inner.update(grads, state.inner_state, params, step)
        updates = _tree_map(lambda u: u * scale, updates)
        return updates, _ScheduledState(inner_state=new_inner, step=state.step + 1)


__all__ = ["Schedule"]
