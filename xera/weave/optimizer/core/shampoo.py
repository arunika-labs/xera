"""Shampoo (Gupta et al. 2018): https://arxiv.org/abs/1802.09568

A practical, simplified full-matrix Shampoo restricted to 2D leaves. For a
weight matrix W of shape (m, n), Shampoo maintains two small preconditioners
instead of one big one: L (m x m, "row" statistics) and R (n x n, "column"
statistics), and preconditions the gradient as `L^(-1/4) @ g @ R^(-1/4)`
(the standard practical approximation to the theoretically-motivated
L^(-1/2) (x) R^(-1/2) Kronecker preconditioner). This is the same shape of
idea as MuonCore's Newton-Schulz orthogonalization -- both replace a plain
per-element learning rate with a whole-matrix-aware direction -- but
Shampoo does it via explicit preconditioner matrices rather than an
orthogonalization iteration.

Leaves with ndim != 2 aren't supported here -- route them elsewhere with
`Partition`, same pattern as MuonCore/Muon:

    opt = O.Partition([
        (lambda path, leaf: leaf.ndim == 2, O.Shampoo(lr=0.01)),
        (lambda path, leaf: True, O.AdamW(lr=1e-4)),
    ])

The inverse 4th-root is computed via eigendecomposition (`jnp.linalg.eigh`),
which is the expensive part of Shampoo -- `precondition_every` controls how
often it's recomputed (every step by default; raise it to amortize the
cost over several steps, reusing the last computed preconditioner in
between, at the cost of slightly staler curvature estimates).

SOAP (Vyas et al. 2024) is a closely related variant -- Shampoo's
preconditioners applied in a rotated (eigenbasis) space -- and is a
plausible future addition here rather than a separate implementation, since
the two share almost all of this machinery.
"""

from __future__ import annotations
from typing import NamedTuple, Any
import jax
import jax.numpy as jnp
from ..base import Optimizer, _tree_map


def _matrix_inv_pth_root(mat, p, eps=1e-6):
    """Symmetric PD matrix M -> M^(-1/p) via eigendecomposition."""
    dim = mat.shape[0]
    reg = mat + eps * jnp.eye(dim, dtype=mat.dtype)
    eigvals, eigvecs = jnp.linalg.eigh(reg)
    eigvals = jnp.maximum(eigvals, eps)
    inv_root = eigvals ** (-1.0 / p)
    return (eigvecs * inv_root) @ eigvecs.T


class ShampooState(NamedTuple):
    step: jnp.ndarray
    L: Any           # per-leaf (m, m) accumulator
    R: Any           # per-leaf (n, n) accumulator
    L_inv_root: Any  # per-leaf cached (m, m) preconditioner
    R_inv_root: Any  # per-leaf cached (n, n) preconditioner
    momentum: Any    # per-leaf (m, n) momentum of the preconditioned grad


class Shampoo(Optimizer):

    def __init__(self, lr, momentum=0.9, beta=1.0, eps=1e-6,
                 precondition_every=1, weight_decay=0.0):
        """
        Args:
            beta: exponential decay for the L/R accumulators. `1.0` (the
                default) matches the original paper's "accumulate forever"
                behavior; set < 1.0 for an exponential-moving-average
                variant that adapts faster to changing curvature.
            precondition_every: recompute the expensive inverse-root every
                N calls; reuses the cached preconditioner in between.
        """
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.eps = eps
        self.precondition_every = int(precondition_every)
        self.weight_decay = weight_decay

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
            L=jax.tree_util.tree_unflatten(treedef, Ls),
            R=jax.tree_util.tree_unflatten(treedef, Rs),
            L_inv_root=jax.tree_util.tree_unflatten(treedef, Lirs),
            R_inv_root=jax.tree_util.tree_unflatten(treedef, Rirs),
            momentum=jax.tree_util.tree_unflatten(treedef, moms),
        )

    def _leaf_update(self, g, L, R, L_ir, R_ir, mom, p, should_recompute):
        new_L = self.beta * L + (1 - self.beta) * (g @ g.T) if self.beta < 1.0 else L + g @ g.T
        new_R = self.beta * R + (1 - self.beta) * (g.T @ g) if self.beta < 1.0 else R + g.T @ g

        def _recompute(_):
            return (
                _matrix_inv_pth_root(new_L, 4.0, self.eps),
                _matrix_inv_pth_root(new_R, 4.0, self.eps),
            )

        def _keep(_):
            return L_ir, R_ir

        new_L_ir, new_R_ir = jax.lax.cond(should_recompute, _recompute, _keep, None)

        precond_g = new_L_ir @ g @ new_R_ir
        new_mom = self.momentum * mom + precond_g

        update = -self.lr * new_mom
        if self.weight_decay and p is not None:
            update = update - self.lr * self.weight_decay * p
        update = jnp.nan_to_num(update)

        return update, new_L, new_R, new_L_ir, new_R_ir, new_mom

    def update(self, grads, state, params=None, step=None):
        should_recompute = (state.step % self.precondition_every) == 0

        leaves_g, treedef = jax.tree_util.tree_flatten(grads)
        leaves_L = treedef.flatten_up_to(state.L)
        leaves_R = treedef.flatten_up_to(state.R)
        leaves_Lir = treedef.flatten_up_to(state.L_inv_root)
        leaves_Rir = treedef.flatten_up_to(state.R_inv_root)
        leaves_mom = treedef.flatten_up_to(state.momentum)
        leaves_p = (
            treedef.flatten_up_to(params) if params is not None
            else [None] * len(leaves_g)
        )

        out_u, out_L, out_R, out_Lir, out_Rir, out_mom = [], [], [], [], [], []
        for g, L, R, Lir, Rir, mom, p in zip(
            leaves_g, leaves_L, leaves_R, leaves_Lir, leaves_Rir, leaves_mom, leaves_p
        ):
            u, nL, nR, nLir, nRir, nmom = self._leaf_update(
                g, L, R, Lir, Rir, mom, p, should_recompute
            )
            out_u.append(u); out_L.append(nL); out_R.append(nR)
            out_Lir.append(nLir); out_Rir.append(nRir); out_mom.append(nmom)

        updates = jax.tree_util.tree_unflatten(treedef, out_u)
        return updates, ShampooState(
            step=state.step + 1,
            L=jax.tree_util.tree_unflatten(treedef, out_L),
            R=jax.tree_util.tree_unflatten(treedef, out_R),
            L_inv_root=jax.tree_util.tree_unflatten(treedef, out_Lir),
            R_inv_root=jax.tree_util.tree_unflatten(treedef, out_Rir),
            momentum=jax.tree_util.tree_unflatten(treedef, out_mom),
        )


__all__ = ["Shampoo", "ShampooState"]
