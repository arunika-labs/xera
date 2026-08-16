"""Cast: dtype casting wrapper for mixed-precision training -- e.g. keep
optimizer state in fp32 while grads/updates flow through the rest of the
training loop in bf16.
"""

from __future__ import annotations
from ..base import Optimizer, _tree_map


class Cast:
    """Factory: casts gradients to `grad_dtype` before the wrapped
    optimizer sees them, and casts its output updates to `update_dtype`
    afterward. Either can be omitted to leave that side uncast.

    Usage:
        opt = O.Cast(grad_dtype="float32", update_dtype="bfloat16")(O.AdamW(lr=1e-3))
    """

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
