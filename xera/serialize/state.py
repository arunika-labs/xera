

"""
State serialization utilities for saving and loading training states.

This module provides functions to save and load training states (including
optimizer states, running statistics, etc.) using a custom pickle-based format
with magic bytes for validation.
"""

from __future__ import annotations
import pickle
import jax
import numpy as np

_MAGIC = b"XERASTAT"
_VERSION = 1


def save_state(state, path):
    """
    Save a training state to a file.
    
    This function serializes a training state (which may include optimizer
    states, running statistics, and other stateful components) to a file
    using pickle with a magic header for validation.
    
    Args:
        state: The state object to save (should be a JAX pytree).
        path: The file path where the state should be saved.
    
    Example:
        >>> save_state(training_state, "training_state.pkl")
    """
    leaves, treedef = jax.tree_util.tree_flatten(state)
    arrays = [np.asarray(leaf) for leaf in leaves]
    with open(path, "wb") as f:
        pickle.dump({"magic": _MAGIC, "version": _VERSION, "treedef": treedef, "arrays": arrays}, f)


def load_state(path):
    """
    Load a training state from a file.
    
    This function loads a training state from a file, validating the magic
    bytes to ensure it's a valid xera state file before attempting to
    reconstruct the state.
    
    Args:
        path: The file path to load the state from.
    
    Returns:
        The reconstructed training state.
    
    Raises:
        ValueError: If the file is not a valid xera state file.
    
    Example:
        >>> training_state = load_state("training_state.pkl")
    """
    with open(path, "rb") as f:
        try:
            blob = pickle.load(f)
        except Exception as e:
            raise ValueError(f"{path} is not a xera state file.") from e
    if not isinstance(blob, dict) or blob.get("magic") != _MAGIC:
        raise ValueError(f"{path} is not a xera state file.")
    leaves = [jax.numpy.asarray(a) for a in blob["arrays"]]
    return jax.tree_util.tree_unflatten(blob["treedef"], leaves)


__all__ = ["save_state", "load_state"]
