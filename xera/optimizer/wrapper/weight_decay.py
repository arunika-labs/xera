

from __future__ import annotations
from typing import NamedTuple, Any
import jax.numpy as jnp
from ..base import Optimizer, _tree_map, _as_hyper, _get_hyper
from ..state import State


class _WeightDecayedState(NamedTuple):
    inner_state: Any
    rate: jnp.ndarray


class WeightDecay(State):
    """
    Adds decoupled weight decay on top of any inner optimizer:
    `updates -= lr * rate * params`, applied after the inner optimizer's
    own update (same decoupled form as `AdamW`'s built-in `weight_decay`,
    just as a wrapper so it composes with optimizers that don't have one
    natively).

    `lr` is *not* configured here -- it's read straight out of the inner
    optimizer's own state after each `update()` call (via the same
    `inner_state`-walking `_get_hyper` that `Schedule` uses), so this
    always uses whatever lr the inner optimizer is actually using that
    step, including a scheduled one. There's therefore only ever one
    source of truth for lr; it can't drift out of sync with the inner
    optimizer, and wrapping in `Schedule` to vary the inner lr "just
    works" here for free.

    If the inner optimizer has no `lr` field at all (e.g. it's some
    custom `Optimizer` that doesn't expose one), pass `lr=` explicitly
    as a fallback constant.
    """

    rate: float = None
    lr: float = None  # fallback only, used if inner has no `lr` leaf

    def __call__(self, inner: Optimizer) -> Optimizer:
        return _WeightDecayed(inner, self.rate, self.lr)


class _WeightDecayed(Optimizer):
    inner: Optimizer = None
    rate: float = None
    lr: float = None  # fallback only

    def init(self, params):
        return _WeightDecayedState(
            inner_state=self.inner.init(params), rate=_as_hyper(self.rate)
        )

    def update(self, grads, state, params=None, step=None):
        updates, new_inner = self.inner.update(grads, state.inner_state, params, step)
        rate = state.rate

        lr = _get_hyper(new_inner, "lr")
        if lr is None:
            if self.lr is None:
                raise TypeError(
                    "WeightDecay: inner optimizer has no `lr` leaf in its "
                    "state, and no fallback `lr=` was given to WeightDecay(...)."
                )
            lr = jnp.asarray(self.lr)

        if params is not None:
            updates = _tree_map(lambda u, p: u - lr * rate * p, updates, params)

        return updates, _WeightDecayedState(inner_state=new_inner, rate=rate)


__all__ = ["WeightDecay"]
