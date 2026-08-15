"""Optimizer base class and helpers shared across every optimizer and
wrapper in this package. Nothing else lives here -- each core optimizer
(SGDMomentum, AdamW, Lion, Muon, ...) and each wrapper (Clip, Schedule,
Accumulate, ...) gets its own file next to this one. See the package's
__init__.py for the full map.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp

_tree_map = jax.tree_util.tree_map


def apply_updates(params, updates):

    return _tree_map(lambda p, u: p + u, params, updates)


def _global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.zeros([])
    return jnp.sqrt(sum(jnp.sum(jnp.square(l)) for l in leaves))


class Optimizer:
    """Base class for all xera.weave optimizers.

    Subclasses implement:
        init(params) -> state
        update(grads, state, params=None, step=None) -> (updates, new_state)

    `step` is an optional externally-threaded step counter -- a scalar the
    *caller's* training loop maintains (e.g. the loop iteration index).
    Passing it explicitly lets every wrapper in a composition (Schedule,
    Accumulate, ...) read the same notion of "step", which avoids an
    ordering ambiguity that otherwise shows up when wrappers are nested and
    each keeps its own internal counter -- see Schedule and Accumulate's
    docstrings for the concrete failure mode. If a caller doesn't pass
    `step` (the default), wrappers that need one fall back to a counter
    they maintain internally in their own state, so everything still works
    without it -- just with that same ambiguity if you nest step-aware
    wrappers.
    """

    def init(self, params):
        raise NotImplementedError

    def update(self, grads, state, params=None, step=None):

        raise NotImplementedError


__all__ = ["apply_updates", "Optimizer"]
