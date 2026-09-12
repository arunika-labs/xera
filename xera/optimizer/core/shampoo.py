

from __future__ import annotations
from typing import NamedTuple, Any
import jax
import jax.numpy as jnp
from ..base import Optimizer, _tree_map, _as_hyper


def _matrix_inv_pth_root(mat, p, eps=1e-6):

    dim = mat.shape[0]
    reg = mat + eps * jnp.eye(dim, dtype=mat.dtype)
    eigvals, eigvecs = jnp.linalg.eigh(reg)
    eigvals = jnp.maximum(eigvals, eps)
    inv_root = eigvals ** (-1.0 / p)
    return (eigvecs * inv_root) @ eigvecs.T


class ShampooState(NamedTuple):
    step: jnp.ndarray
    lr: jnp.ndarray
    momentum: jnp.ndarray
    beta: jnp.ndarray
    eps: jnp.ndarray
    precondition_every: jnp.ndarray
    weight_decay: jnp.ndarray
    L: Any           # per-leaf (m, m) accumulator
    R: Any           # per-leaf (n, n) accumulator
    L_inv_root: Any  # per-leaf cached (m, m) preconditioner
    R_inv_root: Any  # per-leaf cached (n, n) preconditioner
    momentum_buf: Any  # per-leaf (m, n) momentum of the preconditioned grad


class Shampoo(Optimizer):

    lr: float = None
    momentum: float = 0.9
    # Whether L/R fully accumulate (beta==1, no decay) or exponentially
    # decay (beta<1) is structural -- a different arithmetic graph is
    # traced for each branch. Decided once from the constructor value;
    # the actual beta *used inside whichever branch* is still the
    # ordinary dynamic leaf state.beta, so it can still be scheduled.
    beta: float = 1.0
    eps: float = 1e-6
    precondition_every: int = 1
    weight_decay: float = 0.0

    def setup(self):
        self.precondition_every = int(self.precondition_every)

    def init(self, params):
        def _init_leaf(p):
            assert p.ndim == 2, (
                f"Shampoo only supports 2D leaves, got shape {p.shape}. "
                "Route other leaves elsewhere with Partition."
            )
            m, n = p.shape
            return (
                jnp.eye(m, dtype=p.dtype),
                jnp.eye(n, dtype=p.dtype),
                jnp.eye(m, dtype=p.dtype),
                jnp.eye(n, dtype=p.dtype),
                jnp.zeros_like(p),
            )

        leaves, treedef = jax.tree_util.tree_flatten(params)
        Ls, Rs, Lirs, Rirs, moms = [], [], [], [], []
        for p in leaves:
            L, R, Lir, Rir, mom = _init_leaf(p)
            Ls.append(L); Rs.append(R); Lirs.append(Lir); Rirs.append(Rir); moms.append(mom)

        return ShampooState(
            step=jnp.zeros([], jnp.int32),
            lr=_as_hyper(self.lr), momentum=_as_hyper(self.momentum),
            beta=_as_hyper(self.beta), eps=_as_hyper(self.eps),
            precondition_every=_as_hyper(self.precondition_every),
            weight_decay=_as_hyper(self.weight_decay),
            L=jax.tree_util.tree_unflatten(treedef, Ls),
            R=jax.tree_util.tree_unflatten(treedef, Rs),
            L_inv_root=jax.tree_util.tree_unflatten(treedef, Lirs),
            R_inv_root=jax.tree_util.tree_unflatten(treedef, Rirs),
            momentum_buf=jax.tree_util.tree_unflatten(treedef, moms),
        )

    def _leaf_update(self, g, L, R, L_ir, R_ir, mom, p, should_recompute,
                      lr, momentum, beta, eps, weight_decay):
        new_L = beta * L + (1 - beta) * (g @ g.T) if self.beta < 1.0 else L + g @ g.T
        new_R = beta * R + (1 - beta) * (g.T @ g) if self.beta < 1.0 else R + g.T @ g

        def _recompute(_):
            return (
                _matrix_inv_pth_root(new_L, 4.0, eps),
                _matrix_inv_pth_root(new_R, 4.0, eps),
            )

        def _keep(_):
            return L_ir, R_ir

        new_L_ir, new_R_ir = jax.lax.cond(should_recompute, _recompute, _keep, None)

        precond_g = new_L_ir @ g @ new_R_ir
        new_mom = momentum * mom + precond_g

        update = -lr * new_mom
        if p is not None:
            update = update - lr * weight_decay * p
        update = jnp.nan_to_num(update)

        return update, new_L, new_R, new_L_ir, new_R_ir, new_mom

    def update(self, grads, state, params=None, step=None):
        lr, momentum, beta = state.lr, state.momentum, state.beta
        eps, precondition_every = state.eps, state.precondition_every
        weight_decay = state.weight_decay
        should_recompute = (state.step % precondition_every) == 0

        leaves_g, treedef = jax.tree_util.tree_flatten(grads)
        leaves_L = treedef.flatten_up_to(state.L)
        leaves_R = treedef.flatten_up_to(state.R)
        leaves_Lir = treedef.flatten_up_to(state.L_inv_root)
        leaves_Rir = treedef.flatten_up_to(state.R_inv_root)
        leaves_mom = treedef.flatten_up_to(state.momentum_buf)
        leaves_p = (
            treedef.flatten_up_to(params) if params is not None
            else [None] * len(leaves_g)
        )

        out_u, out_L, out_R, out_Lir, out_Rir, out_mom = [], [], [], [], [], []
        for g, L, R, Lir, Rir, mom, p in zip(
            leaves_g, leaves_L, leaves_R, leaves_Lir, leaves_Rir, leaves_mom, leaves_p
        ):
            u, nL, nR, nLir, nRir, nmom = self._leaf_update(
                g, L, R, Lir, Rir, mom, p, should_recompute,
                lr, momentum, beta, eps, weight_decay,
            )
            out_u.append(u); out_L.append(nL); out_R.append(nR)
            out_Lir.append(nLir); out_Rir.append(nRir); out_mom.append(nmom)

        updates = jax.tree_util.tree_unflatten(treedef, out_u)
        return updates, ShampooState(
            step=state.step + 1, lr=lr, momentum=momentum, beta=beta, eps=eps,
            precondition_every=precondition_every, weight_decay=weight_decay,
            L=jax.tree_util.tree_unflatten(treedef, out_L),
            R=jax.tree_util.tree_unflatten(treedef, out_R),
            L_inv_root=jax.tree_util.tree_unflatten(treedef, out_Lir),
            R_inv_root=jax.tree_util.tree_unflatten(treedef, out_Rir),
            momentum_buf=jax.tree_util.tree_unflatten(treedef, out_mom),
        )


__all__ = ["Shampoo", "ShampooState"]
