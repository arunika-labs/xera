"""Accumulate: gradient accumulation wrapper, implemented with jax.lax.cond
so it's fully traceable without a nested loop primitive. Generic -- wraps
any Optimizer, doesn't know or care what it is.
"""

from __future__ import annotations
from typing import NamedTuple, Any
import jax
import jax.numpy as jnp
from ..base import Optimizer, _tree_map


class Accumulate:
    """Factory: buffers `steps` micro-step gradients (summed, then averaged
    on apply) and only runs the wrapped optimizer's update once they've
    accumulated -- every other call emits an all-zero update and just keeps
    buffering. Implemented with `jax.lax.cond`, so it's fully traceable and
    composes with jit/scan/Loop directly; it does not need a nested loop
    primitive.

    Usage:
        opt = O.Accumulate(4)(O.AdamW(lr=1e-4))

    Step source: if an external `step` is threaded in via
    `update(..., step=...)`, Accumulate decides when to apply from that
    value directly (`(step + 1) % steps == 0`) instead of an internal
    counter -- this is the recommended mode, since it guarantees Accumulate
    and any Schedule composed with it agree on what "step" means (see
    Schedule's docstring for the concrete ambiguity this avoids). Without an
    explicit `step`, Accumulate keeps its own internal counter and applies
    every `steps`-th call.
    """

    def __init__(self, steps: int):
        assert steps >= 1, "Accumulate(steps) needs steps >= 1"
        self.steps = int(steps)

    def __call__(self, inner: Optimizer) -> Optimizer:
        return _Accumulated(inner, self.steps)


class _AccumulatedState(NamedTuple):
    inner_state: Any
    buf: Any
    count: jnp.ndarray


class _Accumulated(Optimizer):
    def __init__(self, inner, steps):
        self.inner = inner
        self.steps = steps

    def init(self, params):
        buf = _tree_map(jnp.zeros_like, params)
        return _AccumulatedState(
            inner_state=self.inner.init(params),
            buf=buf,
            count=jnp.zeros([], jnp.int32),
        )

    def update(self, grads, state, params=None, step=None):
        buf = _tree_map(jnp.add, state.buf, grads)
        count = state.count + 1

        if step is None:
            should_apply = (count % self.steps) == 0
        else:
            should_apply = ((jnp.asarray(step) + 1) % self.steps) == 0

        def _apply(operand):
            buf_, inner_state_ = operand
            # Average over the window so the wrapped optimizer sees a
            # gradient at the same scale as an un-accumulated micro-step
            # gradient, not one that's `steps` times too large.
            avg = _tree_map(lambda b: b / self.steps, buf_)
            updates_, new_inner_state_ = self.inner.update(
                avg, inner_state_, params, step
            )
            zero_buf_ = _tree_map(jnp.zeros_like, buf_)
            return updates_, zero_buf_, new_inner_state_

        def _skip(operand):
            buf_, inner_state_ = operand
            zero_updates_ = _tree_map(jnp.zeros_like, buf_)
            return zero_updates_, buf_, inner_state_

        # Both branches must return identical pytree structure/shape/dtype:
        # _apply resets buf to zeros (not drops it), and both branches touch
        # inner_state -- _apply via a real update, _skip via pass-through --
        # so their structures match too.
        updates, new_buf, new_inner_state = jax.lax.cond(
            should_apply, _apply, _skip, (buf, state.inner_state)
        )

        return updates, _AccumulatedState(
            inner_state=new_inner_state, buf=new_buf, count=count
        )


__all__ = ["Accumulate"]
