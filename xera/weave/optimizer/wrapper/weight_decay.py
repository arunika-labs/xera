

from __future__ import annotations
from ..base import Optimizer, _tree_map
from ....core import Struct


def _find_lr(opt):

    seen = set()
    cur = opt
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "lr"):
            return cur.lr
        cur = getattr(cur, "inner", None)
    return None


class WeightDecay(Struct):

    rate: float = None
    lr: float = None

    def setup(self):
        self.rate = float(self.rate)

    def __call__(self, inner: Optimizer) -> Optimizer:
        lr = self.lr if self.lr is not None else _find_lr(inner)
        if lr is None:
            raise TypeError(
                f"WeightDecay could not find a `.lr` attribute on {inner!r} "
                "or anywhere in its wrapped chain. Pass `WeightDecay(rate, "
                "lr=...)` explicitly for optimizers that don't expose `.lr`."
            )
        return _WeightDecayed(inner, self.rate, lr)


class _WeightDecayed(Optimizer):
    inner: Optimizer = None
    rate: float = None
    lr: float = None

    def init(self, params):
        return self.inner.init(params)

    def update(self, grads, state, params=None, step=None):
        updates, new_state = self.inner.update(grads, state, params, step)
        if self.rate and params is not None:
            updates = _tree_map(
                lambda u, p: u - self.lr * self.rate * p, updates, params
            )
        return updates, new_state


__all__ = ["WeightDecay"]
