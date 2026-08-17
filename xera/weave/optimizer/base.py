

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


    def init(self, params):
        raise NotImplementedError

    def update(self, grads, state, params=None, step=None):

        raise NotImplementedError


__all__ = ["apply_updates", "Optimizer"]
