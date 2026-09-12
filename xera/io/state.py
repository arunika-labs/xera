

"""
Optimizer state serialization utilities for saving and loading training
checkpoints.

An `Optimizer`'s state (whatever `.init()`/`.update()` return -- e.g.
`AdamState`, or a wrapped `_WeightDecayedState(inner_state=AdamState(...),
...)`) is a plain JAX pytree of array leaves: `step`, every numeric
hyperparameter (`lr`, `b1`, `weight_decay`, ...), and the optimizer's own
buffers (`m`, `v`, ...) alike. `save_state`/`load_state` serialize that
pytree the same way `save_model`/`load_model` serialize a model's
params -- via safetensors -- so a checkpoint captures training exactly
where it left off, hyperparameters (including any in-progress schedule
value) included, not just the raw moment buffers.
"""

from __future__ import annotations
import jax
import numpy as np
from safetensors.numpy import save_file, load_file
from .model import _key


def save_state(state, path):
    """
    Save an optimizer's state to a safetensors file.

    Flattens `state` (whatever `Optimizer.init()`/`.update()` returned)
    and saves every array leaf -- `step`, every numeric hyperparameter,
    and every buffer (`m`, `v`, `momentum_buf`, ...) alike, at whatever
    depth of wrapper nesting (`Clip`, `WeightDecay`, `Partition`, ...) --
    to a single safetensors file.

    Args:
        state: An optimizer state pytree, as returned by `Optimizer.init()`
            or `Optimizer.update()`.
        path: The file path to save the checkpoint to.

    Example:
        >>> import xera
        >>> import xera.optimizer as xopt
        >>> opt = xopt.Adam(lr=1e-3)
        >>> state = opt.init(params)
        >>> ...  # train for a while
        >>> xera.io.save_state(state, "opt_state.safetensors")
    """
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(state)
    tensors = {_key(p): np.asarray(leaf) for p, leaf in leaves_with_path}
    save_file(tensors, path)


def load_state(template, path):
    """
    Load an optimizer's state from a safetensors file.

    `template` only needs to have the right *shape* -- same optimizer,
    same wrapper chain, same param shapes -- typically just a fresh
    `opt.init(params)` call; every leaf's value gets overwritten from
    the checkpoint, including hyperparameters, so it's fine (and
    expected) for `template`'s hyperparameter values to differ from
    what's in the file, e.g. after changing `Adam(lr=...)`'s constructor
    argument before resuming -- the checkpointed `lr` wins.

    Args:
        template: An optimizer state pytree with the same structure as
            the saved one (same optimizer/wrapper chain, same param
            shapes) -- typically `opt.init(params)` on a freshly
            constructed optimizer.
        path: The file path to load the checkpoint from.

    Returns:
        An optimizer state pytree with the same structure as `template`,
        with every leaf's value loaded from the checkpoint.

    Example:
        >>> import xera
        >>> import xera.optimizer as xopt
        >>> opt = xopt.Adam(lr=1e-3)  # lr= here is only a
        ...                                     # placeholder; the loaded
        ...                                     # checkpoint's lr wins
        >>> state = xera.io.load_state(opt.init(params), "opt_state.safetensors")
    """
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(template)
    tensors = load_file(path)
    leaves = [
        jax.numpy.asarray(tensors[_key(p)]).reshape(leaf.shape).astype(leaf.dtype)
        for p, leaf in leaves_with_path
    ]
    return jax.tree_util.tree_unflatten(treedef, leaves)


__all__ = ["save_state", "load_state"]
