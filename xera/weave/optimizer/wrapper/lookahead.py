

from __future__ import annotations
from typing import NamedTuple, Any
import jax
import jax.numpy as jnp
from ..base import Optimizer, _tree_map


class Lookahead:


    def __init__(self, k: int = 5, alpha: float = 0.5):
        assert k >= 1, "Lookahead(k=...) needs k >= 1"
        self.k = int(k)
        self.alpha = float(alpha)

    def __call__(self, inner: Optimizer) -> Optimizer:
        return _Lookahead(inner, self.k, self.alpha)


class _LookaheadState(NamedTuple):
    inner_state: Any
    slow: Any
    count: jnp.ndarray


class _Lookahead(Optimizer):
    def __init__(self, inner, k, alpha):
        self.inner = inner
        self.k = k
        self.alpha = alpha

    def init(self, params):
        return _LookaheadState(
            inner_state=self.inner.init(params),
            slow=params,
            count=jnp.zeros([], jnp.int32),
        )

    def update(self, grads, state, params=None, step=None):
        if params is None:
            raise ValueError(
                "Lookahead needs `params` (not just grads) to compute the "
                "fast/slow interpolation -- pass params to update()."
            )

        fast_updates, new_inner = self.inner.update(grads, state.inner_state, params, step)
        fast_params = _tree_map(lambda p, u: p + u, params, fast_updates)

        count = state.count + 1
        if step is None:
            should_sync = (count % self.k) == 0
        else:
            should_sync = ((jnp.asarray(step) + 1) % self.k) == 0

        def _sync(operand):
            slow_, fast_ = operand
            new_slow = _tree_map(
                lambda s, f: s + self.alpha * (f - s), slow_, fast_
            )
            return new_slow, new_slow  # fast resets to the newly-synced slow point

        def _no_sync(operand):
            slow_, fast_ = operand
            return slow_, fast_

        new_slow, new_fast = jax.lax.cond(
            should_sync, _sync, _no_sync, (state.slow, fast_params)
        )

        # `updates` must be a delta from the *original* params passed in,
        # not from fast_params -- apply_updates(params, updates) needs to
        # land on new_fast regardless of whether a sync happened this call.
        updates = _tree_map(lambda nf, p: nf - p, new_fast, params)

        return updates, _LookaheadState(
            inner_state=new_inner, slow=new_slow, count=count
        )


__all__ = ["Lookahead"]
