

from __future__ import annotations
import pickle
import jax
import numpy as np

_MAGIC = b"XERASTAT"
_VERSION = 1


def save_state(state, path):
    """Serialize any xera.weave state pytree (optimizer state, Train/Loop
    state, EMA buffers, resume counters, ...) to `path`. Unlike
    `xera.serialize.model`, this covers arbitrary pytree structure
    (NamedTuples, ints, nested Optimizer state), not just arrays -- so it
    is xera's own format rather than safetensors. Only meant for state you
    produced yourself (e.g. to resume training), not for distributing
    weights.
    """
    leaves, treedef = jax.tree_util.tree_flatten(state)
    arrays = [np.asarray(leaf) for leaf in leaves]
    with open(path, "wb") as f:
        pickle.dump({"magic": _MAGIC, "version": _VERSION, "treedef": treedef, "arrays": arrays}, f)


def load_state(path):
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
