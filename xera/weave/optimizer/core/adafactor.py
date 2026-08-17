

from __future__ import annotations
from typing import NamedTuple, Any
import jax
import jax.numpy as jnp
from ..base import Optimizer, _tree_map


class AdafactorState(NamedTuple):
    step: jnp.ndarray
    v_row: Any   # per-leaf: row factor (shape (m,)) for 2D leaves, else None marker unused
    v_col: Any   # per-leaf: col factor (shape (n,)) for 2D leaves
    v_full: Any  # per-leaf: full second moment for non-2D leaves


# Small values used as the "not applicable" slot for a leaf's unused
# factorization arm, so every leaf position has a fixed-shape/dtype value
# in both v_row/v_col/v_full trees regardless of that leaf's ndim -- this
# keeps the whole state a single consistent pytree instead of a Python
# per-leaf branch that would confuse tree_map.
_UNUSED = jnp.zeros([1])


class Adafactor(Optimizer):

    def __init__(self, lr, decay=0.8, eps=1e-30, clip_threshold=1.0, weight_decay=0.0):
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.clip_threshold = clip_threshold
        self.weight_decay = weight_decay

    def _init_leaf(self, p):
        if p.ndim == 2:
            m, n = p.shape
            return jnp.zeros((m,)), jnp.zeros((n,)), _UNUSED
        else:
            return _UNUSED, _UNUSED, jnp.zeros_like(p)

    def init(self, params):
        leaves, treedef = jax.tree_util.tree_flatten(params)
        rows, cols, fulls = [], [], []
        for p in leaves:
            r, c, f = self._init_leaf(p)
            rows.append(r)
            cols.append(c)
            fulls.append(f)
        return AdafactorState(
            step=jnp.zeros([], jnp.int32),
            v_row=jax.tree_util.tree_unflatten(treedef, rows),
            v_col=jax.tree_util.tree_unflatten(treedef, cols),
            v_full=jax.tree_util.tree_unflatten(treedef, fulls),
        )

    def _leaf_update(self, g, v_row, v_col, v_full, p):
        decay = self.decay
        if g.ndim == 2:
            row_mean = jnp.mean(jnp.square(g), axis=1) + self.eps
            col_mean = jnp.mean(jnp.square(g), axis=0) + self.eps
            new_v_row = decay * v_row + (1 - decay) * row_mean
            new_v_col = decay * v_col + (1 - decay) * col_mean
            # Rank-1 reconstruction of the full second-moment estimate from
            # the row/column factors (the whole point of factoring).
            r = new_v_row / jnp.mean(new_v_row)
            approx_v = jnp.outer(r, new_v_col)
            denom = jnp.sqrt(approx_v) + self.eps
            direction = g / denom
            new_v_full = v_full  # unused arm stays as-is
        else:
            new_v_full = decay * v_full + (1 - decay) * jnp.square(g)
            denom = jnp.sqrt(new_v_full) + self.eps
            direction = g / denom
            new_v_row, new_v_col = v_row, v_col

        # Update clipping: rescale so the RMS of `direction` doesn't exceed
        # clip_threshold, as in the original paper -- keeps early, noisy
        # second-moment estimates from producing huge steps.
        if self.clip_threshold is not None:
            rms = jnp.sqrt(jnp.mean(jnp.square(direction)))
            direction = direction / jnp.maximum(1.0, rms / self.clip_threshold)

        update = -self.lr * direction
        if self.weight_decay and p is not None:
            update = update - self.lr * self.weight_decay * p

        return update, new_v_row, new_v_col, new_v_full

    def update(self, grads, state, params=None, step=None):
        leaves_g, treedef = jax.tree_util.tree_flatten(grads)
        leaves_vr = treedef.flatten_up_to(state.v_row)
        leaves_vc = treedef.flatten_up_to(state.v_col)
        leaves_vf = treedef.flatten_up_to(state.v_full)
        leaves_p = (
            treedef.flatten_up_to(params) if params is not None
            else [None] * len(leaves_g)
        )

        out_u, out_vr, out_vc, out_vf = [], [], [], []
        for g, vr, vc, vf, p in zip(leaves_g, leaves_vr, leaves_vc, leaves_vf, leaves_p):
            u, nvr, nvc, nvf = self._leaf_update(g, vr, vc, vf, p)
            out_u.append(u)
            out_vr.append(nvr)
            out_vc.append(nvc)
            out_vf.append(nvf)

        updates = jax.tree_util.tree_unflatten(treedef, out_u)
        return updates, AdafactorState(
            step=state.step + 1,
            v_row=jax.tree_util.tree_unflatten(treedef, out_vr),
            v_col=jax.tree_util.tree_unflatten(treedef, out_vc),
            v_full=jax.tree_util.tree_unflatten(treedef, out_vf),
        )


__all__ = ["Adafactor", "AdafactorState"]
