

from __future__ import annotations
from typing import NamedTuple, Any
import jax
import jax.numpy as jnp
from ..base import Optimizer, _tree_map
from .adam import AdamW
from .lion import Lion
from .sgd import SGDMomentum
from ..partition import Partition


def _newton_schulz5(x, steps=5, eps=1e-7):

    a, b, c = 3.4445, -4.7750, 2.0315
    x = x / (jnp.linalg.norm(x) + eps)

    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    def body(x, _):
        A = x @ x.T
        B = b * A + c * (A @ A)
        return a * x + B @ x, None

    x, _ = jax.lax.scan(body, x, None, length=steps)

    if transposed:
        x = x.T
    return x


class MuonCoreState(NamedTuple):
    step: jnp.ndarray
    momentum: Any


class MuonCore(Optimizer):

    lr: float = None
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    weight_decay: float = 0.0
    clip: bool = True

    def setup(self):
        clip = self.clip
        if clip is True:
            self.clip_threshold = 1.0
        elif clip is False:
            self.clip_threshold = None
        else:
            self.clip_threshold = float(clip)

    def _leaf_update(self, g, m, p):
        new_m = self.momentum * m + g
        direction = g + self.momentum * new_m if self.nesterov else new_m

        if self.clip_threshold is not None:
            norm = jnp.sqrt(jnp.sum(jnp.square(direction)))
            direction = direction * jnp.minimum(
                1.0, self.clip_threshold / (norm + 1e-7)
            )

        if direction.ndim == 2:
            o = _newton_schulz5(direction, steps=self.ns_steps)
            scale = self.lr * jnp.sqrt(
                jnp.maximum(1.0, direction.shape[0] / direction.shape[1])
            )
            update = -scale * o
        elif direction.ndim == 3:
            # Batched matrices, e.g. per-head QK weights of shape (H, D_in, D_out).
            o = jax.vmap(lambda d: _newton_schulz5(d, steps=self.ns_steps))(direction)
            scale = self.lr * jnp.sqrt(
                jnp.maximum(1.0, direction.shape[1] / direction.shape[2])
            )
            update = -scale * o
        elif direction.ndim == 4:
            # Conv kernel (out_ch, in_ch, kh, kw): flatten to 2D, orthogonalize,
            # reshape back -- the standard trick for using Muon on conv layers.
            out_ch = direction.shape[0]
            flat = direction.reshape(out_ch, -1)
            o = _newton_schulz5(flat, steps=self.ns_steps)
            scale = self.lr * jnp.sqrt(jnp.maximum(1.0, flat.shape[0] / flat.shape[1]))
            update = (-scale * o).reshape(direction.shape)
        else:
            update = -self.lr * direction

        if self.weight_decay and p is not None:
            update = update - self.lr * self.weight_decay * p

        # Safety net: NS iterations aren't globally convergent, so a bad
        # gradient can in principle still blow up. Never let a NaN reach params.
        update = jnp.nan_to_num(update)
        return update, new_m

    def init(self, params):
        m = _tree_map(jnp.zeros_like, params)
        return MuonCoreState(step=jnp.zeros([], jnp.int32), momentum=m)

    def update(self, grads, state, params=None, step=None):
        leaves_g, treedef = jax.tree_util.tree_flatten(grads)
        leaves_m = treedef.flatten_up_to(state.momentum)
        leaves_p = (
            treedef.flatten_up_to(params) if params is not None
            else [None] * len(leaves_g)
        )

        out_u, out_m = [], []
        for g, m, p in zip(leaves_g, leaves_m, leaves_p):
            u, nm = self._leaf_update(g, m, p)
            out_u.append(u)
            out_m.append(nm)

        updates = jax.tree_util.tree_unflatten(treedef, out_u)
        new_momentum = jax.tree_util.tree_unflatten(treedef, out_m)
        return updates, MuonCoreState(step=state.step + 1, momentum=new_momentum)


def Muon(
    lr,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
    weight_decay=0.0,
    clip=True,
    fallback="adamw",
    fallback_lr=1e-4,
    include=None,
):

    core = MuonCore(
        lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
        weight_decay=weight_decay, clip=clip,
    )

    if include is not None:
        is_muon_leaf = include
    else:
        def is_muon_leaf(path, leaf):
            return hasattr(leaf, "ndim") and leaf.ndim in (2, 3, 4)

    rules = [(is_muon_leaf, core)]

    if fallback is not None:
        if isinstance(fallback, Optimizer):
            fallback_opt = fallback
        elif fallback == "adamw":
            fallback_opt = AdamW(lr=fallback_lr)
        elif fallback == "lion":
            fallback_opt = Lion(lr=fallback_lr)
        elif fallback == "sgd":
            fallback_opt = SGDMomentum(lr=fallback_lr)
        else:
            raise ValueError(
                f"unknown fallback: {fallback!r} "
                "(expected 'adamw', 'lion', 'sgd', an Optimizer instance, or None)"
            )
        rules.append((lambda path, leaf: True, fallback_opt))

    return Partition(rules)


__all__ = ["MuonCore", "MuonCoreState", "Muon"]
