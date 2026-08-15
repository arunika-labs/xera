

from __future__ import annotations
from typing import NamedTuple, Any, Optional
import jax
import jax.numpy as jnp

_tree_map = jax.tree_util.tree_map


def apply_updates(params, updates):
    
    return _tree_map(lambda p, u: p + u, params, updates)


class Optimizer:
    

    def init(self, params):
        raise NotImplementedError

    def update(self, grads, state, params=None):
        
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SGD with (optionally Nesterov) momentum
# ---------------------------------------------------------------------------

class SGDMomentumState(NamedTuple):
    step: jnp.ndarray
    momentum: Any


class SGDMomentum(Optimizer):
    

    def __init__(self, lr, momentum=0.9, nesterov=False, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay

    def init(self, params):
        m = _tree_map(jnp.zeros_like, params)
        return SGDMomentumState(step=jnp.zeros([], jnp.int32), momentum=m)

    def update(self, grads, state, params=None):
        if self.weight_decay and params is not None:
            grads = _tree_map(
                lambda g, p: g + self.weight_decay * p, grads, params
            )

        new_m = _tree_map(
            lambda m, g: self.momentum * m + g, state.momentum, grads
        )

        if self.nesterov:
            direction = _tree_map(
                lambda m, g: self.momentum * m + g, new_m, grads
            )
        else:
            direction = new_m

        updates = _tree_map(lambda d: -self.lr * d, direction)
        return updates, SGDMomentumState(step=state.step + 1, momentum=new_m)


# ---------------------------------------------------------------------------
# AdamW (decoupled weight decay)
# ---------------------------------------------------------------------------

class AdamWState(NamedTuple):
    step: jnp.ndarray
    m: Any
    v: Any


class AdamW(Optimizer):
    

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay

    def init(self, params):
        m = _tree_map(jnp.zeros_like, params)
        v = _tree_map(jnp.zeros_like, params)
        return AdamWState(step=jnp.zeros([], jnp.int32), m=m, v=v)

    def update(self, grads, state, params=None):
        step = state.step + 1
        step_f = step.astype(jnp.float32)

        m = _tree_map(lambda m, g: self.b1 * m + (1 - self.b1) * g, state.m, grads)
        v = _tree_map(
            lambda v, g: self.b2 * v + (1 - self.b2) * jnp.square(g), state.v, grads
        )

        bias_c1 = 1 - self.b1 ** step_f
        bias_c2 = 1 - self.b2 ** step_f

        updates = _tree_map(
            lambda m, v: -self.lr * (m / bias_c1) / (jnp.sqrt(v / bias_c2) + self.eps),
            m, v,
        )

        # Decoupled weight decay: applied directly to params, not folded into grads.
        if self.weight_decay and params is not None:
            updates = _tree_map(
                lambda u, p: u - self.lr * self.weight_decay * p, updates, params
            )

        return updates, AdamWState(step=step, m=m, v=v)


# ---------------------------------------------------------------------------
# Lion (Evolved Sign Momentum)
# ---------------------------------------------------------------------------

class LionState(NamedTuple):
    step: jnp.ndarray
    m: Any


class Lion(Optimizer):
    

    def __init__(self, lr, b1=0.9, b2=0.99, weight_decay=0.0):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay

    def init(self, params):
        m = _tree_map(jnp.zeros_like, params)
        return LionState(step=jnp.zeros([], jnp.int32), m=m)

    def update(self, grads, state, params=None):
        direction = _tree_map(
            lambda m, g: jnp.sign(self.b1 * m + (1 - self.b1) * g),
            state.m, grads,
        )
        new_m = _tree_map(
            lambda m, g: self.b2 * m + (1 - self.b2) * g, state.m, grads
        )

        updates = _tree_map(lambda d: -self.lr * d, direction)

        if self.weight_decay and params is not None:
            updates = _tree_map(
                lambda u, p: u - self.lr * self.weight_decay * p, updates, params
            )

        return updates, LionState(step=state.step + 1, m=new_m)


# ---------------------------------------------------------------------------
# Muon (MomentUm Orthogonalized by Newton-schulz)
#
# Orthogonalizes the momentum of matrix-shaped leaves via a quintic
# Newton-Schulz iteration (coefficients from Keller Jordan's original Muon:
# https://kellerjordan.github.io/posts/muon/). Leaves that aren't selected
# (by `include`, or by the default ndim-based rule) are routed to a
# `fallback` optimizer instead -- this is required, since Muon is only
# well-defined for matrix-shaped params (hidden Dense/QK-style weights),
# never for scalars, biases, gains, or embeddings.
# ---------------------------------------------------------------------------

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


class MuonState:
    

    __slots__ = ("step", "momentum", "fallback_state", "mask")

    def __init__(self, step, momentum, fallback_state, mask):
        self.step = step
        self.momentum = momentum
        self.fallback_state = fallback_state
        self.mask = mask

    def __repr__(self):
        return f"MuonState(step={self.step!r}, mask={self.mask!r})"


jax.tree_util.register_pytree_node(
    MuonState,
    lambda s: ((s.step, s.momentum, s.fallback_state), s.mask),
    lambda mask, children: MuonState(children[0], children[1], children[2], mask),
)


class Muon(Optimizer):
    

    def __init__(
        self,
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
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
        self.include = include

        if clip is True:
            self.clip_threshold = 1.0
        elif clip is False:
            self.clip_threshold = None
        else:
            self.clip_threshold = float(clip)

        if isinstance(fallback, Optimizer):
            self.fallback_opt = fallback
        elif fallback == "adamw":
            self.fallback_opt = AdamW(lr=fallback_lr)
        elif fallback == "lion":
            self.fallback_opt = Lion(lr=fallback_lr)
        elif fallback == "sgd":
            self.fallback_opt = SGDMomentum(lr=fallback_lr)
        else:
            raise ValueError(
                f"unknown fallback: {fallback!r} "
                "(expected 'adamw', 'lion', 'sgd', or an Optimizer instance)"
            )

    def _select(self, path, leaf):
        if not hasattr(leaf, "ndim"):
            return False
        if self.include is not None:
            return bool(self.include(path, leaf))
        # Default: orthogonalize anything matrix-shaped (2D dense/QK weights,
        # 3D batched/per-head weights, 4D conv kernels). Everything else
        # (scalars, biases, 1D gains) goes to the fallback optimizer.
        return leaf.ndim in (2, 3, 4)

    def _muon_leaf_update(self, g, m, p):
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
        path_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
        mask = tuple(self._select(p, l) for p, l in path_leaves)
        leaves = [l for _, l in path_leaves]

        momentum = tuple(
            jnp.zeros_like(l) if m else None for l, m in zip(leaves, mask)
        )
        fallback_leaves = tuple(l for l, m in zip(leaves, mask) if not m)
        fallback_state = self.fallback_opt.init(fallback_leaves)

        return MuonState(
            step=jnp.zeros([], jnp.int32),
            momentum=momentum,
            fallback_state=fallback_state,
            mask=mask,
        )

    def update(self, grads, state, params=None):
        path_leaves, treedef = jax.tree_util.tree_flatten_with_path(grads)
        grad_leaves = [l for _, l in path_leaves]
        if params is not None:
            param_leaves = treedef.flatten_up_to(params)
        else:
            param_leaves = [None] * len(grad_leaves)

        mask = state.mask
        muon_updates = [None] * len(grad_leaves)
        new_momentum = list(state.momentum)

        for i, (g, p, is_muon) in enumerate(zip(grad_leaves, param_leaves, mask)):
            if is_muon:
                u, new_m = self._muon_leaf_update(g, state.momentum[i], p)
                muon_updates[i] = u
                new_momentum[i] = new_m

        fallback_grads = tuple(g for g, is_muon in zip(grad_leaves, mask) if not is_muon)
        fallback_params = (
            tuple(p for p, is_muon in zip(param_leaves, mask) if not is_muon)
            if params is not None else None
        )
        fallback_updates, new_fallback_state = self.fallback_opt.update(
            fallback_grads, state.fallback_state, fallback_params
        )

        fb_iter = iter(fallback_updates)
        for i, is_muon in enumerate(mask):
            if not is_muon:
                muon_updates[i] = next(fb_iter)

        updates = jax.tree_util.tree_unflatten(treedef, muon_updates)
        new_state = MuonState(
            step=state.step + 1,
            momentum=tuple(new_momentum),
            fallback_state=new_fallback_state,
            mask=mask,
        )
        return updates, new_state


__all__ = [
    "Optimizer",
    "apply_updates",
    "SGDMomentum",
    "AdamW",
    "Lion",
    "Muon",
]
