

"""
State serialization utilities for saving and loading training states.

This module provides functions to save and load training states (optimizer
states, `xera.weave.Train`/`State` instances, running metrics, step
counters, and other stateful pytrees) using the safetensors format --
mirroring `xera.serialize.model`.

There is no pickle involved and no arbitrary object is ever written to
disk. Just like `save_model`/`load_model`, this follows a *template*
pattern: the on-disk file only ever holds the dynamic pytree leaves (as
named tensors), never the tree structure itself. To load a state back you
provide a freshly constructed `template` with the same shape (e.g. call
`optimizer.init(params)` again, or build the same `Train(...)` instance) --
exactly how you already provide an empty model with the right architecture
to `load_model`. Static configuration (learning rates, `loop_type`, the
`Optimizer` object itself, callables, etc.) intentionally never touches the
file: it lives in your code, not in the checkpoint.
"""

from __future__ import annotations
import jax
import numpy as np
from safetensors.numpy import save_file, load_file
from .model import _key


def save_state(state, path):
    """
    Save a training state's dynamic pytree leaves to a safetensors file.

    This flattens `state` (an optimizer state, a `Train`/`State` instance,
    a plain dict of arrays, or any other JAX pytree) and writes every leaf
    as a named tensor -- the same mechanism `save_model` uses for module
    parameters. Static/config attributes of `State` subclasses are not
    leaves and are therefore not written; reconstruct them via `template`
    when loading.

    Args:
        state: The state object to save (any JAX pytree).
        path: The file path where the state should be saved.

    Example:
        >>> save_state(opt_state, "opt_state.safetensors")
    """
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(state)
    tensors = {_key(p): np.asarray(leaf) for p, leaf in leaves_with_path}
    save_file(tensors, path)


def load_state(template, path):
    """
    Load a training state from a safetensors file.

    Loads leaves from a safetensors file and reconstructs a state using
    `template` to determine the pytree structure (including static
    config, such as an `Optimizer`'s hyperparameters or a `Train`'s
    `loop_type`) -- the same way `load_model` uses an empty model as an
    architecture template.

    Args:
        template: A state with the same structure as the saved state
            (e.g. the result of calling `optimizer.init(params)` again).
        path: The file path to load the state from.

    Returns:
        A reconstructed state with `template`'s structure/config and the
        saved leaf values.

    Example:
        >>> template = optimizer.init(params)
        >>> state = load_state(template, "opt_state.safetensors")
    """
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(template)
    tensors = load_file(path)
    leaves = [
        jax.numpy.asarray(tensors[_key(p)]).reshape(leaf.shape).astype(leaf.dtype)
        for p, leaf in leaves_with_path
    ]
    return jax.tree_util.tree_unflatten(treedef, leaves)


__all__ = ["save_state", "load_state"]
