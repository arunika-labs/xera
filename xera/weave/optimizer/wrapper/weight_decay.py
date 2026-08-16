"""WeightDecay: generic decoupled weight decay wrapper, for optimizers
that don't have decay built in (or when you want it applied outside a
composition rather than baked into a specific core optimizer).
"""

from __future__ import annotations
from ..base import Optimizer, _tree_map


def _find_lr(opt):
    """Walk an optimizer's `.inner` chain looking for a `.lr` attribute.
    Core optimizers expose `.lr` directly; a wrapper (Clip, Schedule, ...)
    exposes `.inner`, so this finds the lr of whatever core optimizer is
    at the bottom of the chain.
    """
    seen = set()
    cur = opt
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "lr"):
            return cur.lr
        cur = getattr(cur, "inner", None)
    return None


class WeightDecay:
    """Factory: applies decoupled weight decay -- `updates -= lr * rate *
    params` -- after the wrapped optimizer's own update, the same way
    AdamW/Lion/SGDMomentum/MuonCore apply their own built-in
    `weight_decay` argument. Useful for optimizers without built-in decay
    (e.g. `Shampoo`, plain `Adam`), or to keep decay as an explicit,
    separately-configured step in a composition rather than a constructor
    argument buried inside one optimizer.

    Usage:
        opt = O.WeightDecay(0.01)(O.Shampoo(lr=0.01))

    `lr` is looked up automatically from the wrapped optimizer (it walks
    through any wrappers already applied, e.g.
    `WeightDecay(0.01)(Clip(1.0)(AdamW(lr=1e-3)))` finds AdamW's `.lr`
    through Clip). If the wrapped optimizer doesn't expose `.lr` anywhere
    in its chain (e.g. a fully custom optimizer), pass `lr=` explicitly.
    """

    def __init__(self, rate: float, lr: float = None):
        self.rate = float(rate)
        self.lr = lr

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
    def __init__(self, inner, rate, lr):
        self.inner = inner
        self.rate = rate
        self.lr = lr

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
