

"""
State serialization utilities for saving and loading training states.

This module provides functions to save and load training states (optimizer
states, `xera.weave.Struct` instances, running metrics, step
counters, and other stateful pytrees) using the safetensors format --
mirroring `xera.serialize.model`.

There is no pickle involved and no arbitrary object is ever written to
disk. Just like `save_model`/`load_model`, this follows a *template*
pattern: the on-disk file only ever holds the dynamic pytree leaves (as
named tensors), never the tree structure itself. To load a state back you
provide a freshly constructed `template` with the same shape (e.g. call
`optimizer.init(params)` again, or build the same `Struct(...)` instance) --
exactly how you already provide an empty model with the right architecture
to `load_model`. Static configuration (learning rates, `loop_type`, the
`Optimizer` object itself, callables, etc.) intentionally never touches the
file: it lives in your code, not in the checkpoint.

`save_state` additionally stamps the saved pytree's structure (its JAX
`treedef`, which for `Struct`/`Module` instances includes their static
config -- hyperparameters, `loop_type`, and the like) into the
safetensors file's metadata header. This is purely informative, exactly
like any other safetensors metadata: it does not affect the tensors and
the file is still a plain, directly-loadable safetensors file. `load_state`
uses that stamp to detect drift between the `template` you pass in and
the structure the state was actually saved with.

- `release=False` (default): if `template`'s treedef differs from the
  stamped one (e.g. you changed a hyperparameter, or `Struct` field, since
  the checkpoint was saved), `load_state` raises `ValueError`. This is the
  safety rail: an unintentional config change won't be silently loaded.
- `release=True`: the drift check is skipped and `template`'s structure/
  config is used as-is, on the assumption the change is intentional (e.g.
  releasing a checkpoint under updated hyperparameters).
- Older files with no stamped metadata (or a `template` that isn't a
  registered pytree with static config) always load like before, since
  there is nothing to compare against.
"""

from __future__ import annotations
import jax
import numpy as np
from safetensors.numpy import save_file, load_file
from safetensors import safe_open
from .model import _key

_METADATA_TREEDEF_KEY = "xera_treedef"


def save_state(state, path):
    """
    Save a training state's dynamic pytree leaves to a safetensors file.

    This flattens `state` (an optimizer state, a `Struct` instance,
    a plain dict of arrays, or any other JAX pytree) and writes every leaf
    as a named tensor -- the same mechanism `save_model` uses for module
    parameters. Static/config attributes of `Struct` subclasses are not
    leaves and are therefore not written as tensors; reconstruct them via
    `template` when loading. A string form of the pytree's structure
    (including that static config) is stamped into the file's metadata
    header so `load_state` can later detect config drift.

    Args:
        state: The state object to save (any JAX pytree).
        path: The file path where the state should be saved.

    Example:
        >>> save_state(opt_state, "opt_state.safetensors")
    """
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(state)
    tensors = {_key(p): np.asarray(leaf) for p, leaf in leaves_with_path}
    save_file(tensors, path, metadata={_METADATA_TREEDEF_KEY: repr(treedef)})


def load_state(template, path, release=False):
    """
    Load a training state from a safetensors file.

    Loads leaves from a safetensors file and reconstructs a state using
    `template` to determine the pytree structure (including static
    config, such as an `Optimizer`'s hyperparameters) -- the same way
    `load_model` uses an empty model as an architecture template.

    Args:
        template: A state with the same structure as the saved state
            (e.g. the result of calling `optimizer.init(params)` again,
            or a freshly constructed `Struct` instance).
        path: The file path to load the state from.
        release: If `False` (default), a mismatch between `template`'s
            structure/static config and the structure the file was saved
            with raises `ValueError` -- treating the mismatch as an
            unintentional config change. If `True`, the mismatch is
            treated as an intentional change (e.g. releasing a checkpoint
            under new hyperparameters): the check is skipped and
            `template`'s structure/config is used.

    Returns:
        A reconstructed state with `template`'s structure/config and the
        saved leaf values.

    Raises:
        ValueError: If `release=False` and `template`'s structure/static
            config does not match what was stamped into `path` at save
            time.

    Example:
        >>> template = optimizer.init(params)
        >>> state = load_state(template, "opt_state.safetensors")
        >>> # Intentionally changed a hyperparameter since saving:
        >>> state = load_state(new_template, "opt_state.safetensors", release=True)
    """
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(template)

    if not release:
        with safe_open(path, framework="numpy") as f:
            saved_treedef_repr = f.metadata().get(_METADATA_TREEDEF_KEY) if f.metadata() else None
        if saved_treedef_repr is not None and saved_treedef_repr != repr(treedef):
            raise ValueError(
                "load_state: template's structure/config doesn't match the "
                "checkpoint at '{}'.\n  saved:    {}\n  template: {}\n"
                "If this change is intentional, pass release=True.".format(
                    path, saved_treedef_repr, repr(treedef)
                )
            )

    tensors = load_file(path)
    leaves = [
        jax.numpy.asarray(tensors[_key(p)]).reshape(leaf.shape).astype(leaf.dtype)
        for p, leaf in leaves_with_path
    ]
    return jax.tree_util.tree_unflatten(treedef, leaves)


__all__ = ["save_state", "load_state"]
