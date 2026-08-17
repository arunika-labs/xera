

from __future__ import annotations
from ..base import Optimizer, _tree_map


class Cast:


    def __init__(self, grad_dtype=None, update_dtype=None):
        self.grad_dtype = grad_dtype
        self.update_dtype = update_dtype

    def __call__(self, inner: Optimizer) -> Optimizer:
        return _Cast(inner, self.grad_dtype, self.update_dtype)


class _Cast(Optimizer):
    def __init__(self, inner, grad_dtype, update_dtype):
        self.inner = inner
        self.grad_dtype = grad_dtype
        self.update_dtype = update_dtype

    def init(self, params):
        return self.inner.init(params)

    def update(self, grads, state, params=None, step=None):
        if self.grad_dtype is not None:
            grads = _tree_map(lambda g: g.astype(self.grad_dtype), grads)

        updates, new_state = self.inner.update(grads, state, params, step)

        if self.update_dtype is not None:
            updates = _tree_map(lambda u: u.astype(self.update_dtype), updates)

        return updates, new_state


__all__ = ["Cast"]
