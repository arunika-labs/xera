

from __future__ import annotations
import jax
import jax.numpy as jnp
from .state import State

_tree_map = jax.tree_util.tree_map


def apply_updates(params, updates):

    return _tree_map(lambda p, u: p + u, params, updates)


def _global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.zeros([])
    return jnp.sqrt(sum(jnp.sum(jnp.square(l)) for l in leaves))


def _as_hyper(value):
    """
    Normalize a hyperparameter's constructor value into a `jnp.ndarray`.

    Every numeric hyperparameter (`lr`, `b1`, `eps`, `weight_decay`,
    ...) lives as a leaf in the state a core optimizer returns -- same
    as `m`/`v`/etc -- rather than as static config on the `Optimizer`
    object. `init()` calls this once per hyperparameter to seed that
    leaf from the constructor value; `update()` then reads the
    *current* value from `state.<name>` (not `self.<name>`) and by
    default just carries it forward unchanged.

    This is what makes hyperparameters checkpoint-visible like params,
    and what lets a wrapper like `Schedule` vary one of them over
    training by overwriting its leaf in `state` before delegating to
    the wrapped optimizer -- without ever touching `Optimizer` config,
    so it can never trigger a recompile.

    Structural/control-flow hyperparameters (bools gating a Python
    `if`, ints sizing a `lax.scan`) are NOT numeric hyperparameters in
    this sense -- they stay static `self.<name>` config, since JAX
    needs their concrete value at trace time.
    """
    return jnp.asarray(value)


def _get_hyper(state, name):
    """
    Find a named hyperparameter leaf in `state`, recursing into
    `state.inner_state` for wrapped optimizers.

    Returns `None` if no field named `name` is found anywhere in the
    chain (e.g. asking for `weight_decay` on an optimizer that doesn't
    have one).
    """
    if hasattr(state, name):
        return getattr(state, name)
    inner = getattr(state, "inner_state", None)
    if inner is not None:
        return _get_hyper(inner, name)
    return None


def _set_hyper(state, name, value):
    """
    Return a copy of `state` with the named hyperparameter leaf
    overwritten, recursing into `state.inner_state` for wrapped
    optimizers.

    Raises `AttributeError` if no field named `name` exists anywhere
    in the chain -- unlike `_get_hyper`, silently no-oping here would
    make a `Schedule(fn, target="typo")` fail invisibly.
    """
    if hasattr(state, name):
        return state._replace(**{name: jnp.asarray(value)})
    inner = getattr(state, "inner_state", None)
    if inner is not None:
        return state._replace(inner_state=_set_hyper(inner, name, value))
    raise AttributeError(
        f"no hyperparameter {name!r} found in this optimizer's state chain"
    )


class Optimizer(State):


    def init(self, params):
        raise NotImplementedError

    def update(self, grads, state, params=None, step=None):

        raise NotImplementedError


__all__ = [
    "apply_updates",
    "Optimizer",
    "_as_hyper",
    "_get_hyper",
    "_set_hyper",
]
