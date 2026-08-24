

from __future__ import annotations
from ..base import Optimizer, _tree_map
from ....core import Struct


class Cast(Struct):

    grad_dtype: object = None
    update_dtype: object = None

    def __call__(self, inner: Optimizer) -> Optimizer:
        return _Cast(inner, self.grad_dtype, self.update_dtype)


class _Cast(Optimizer):
    inner: Optimizer = None
    grad_dtype: object = None
    update_dtype: object = None

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
