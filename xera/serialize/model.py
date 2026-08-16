

from __future__ import annotations
import jax
import numpy as np
from safetensors.numpy import save_file, load_file


def _key(path):
    return jax.tree_util.keystr(path).lstrip(".")


def save_model(module, path):
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(module)
    tensors = {_key(p): np.asarray(leaf) for p, leaf in leaves_with_path}
    save_file(tensors, path)


def load_model(template, path):
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(template)
    tensors = load_file(path)
    leaves = [
        jax.numpy.asarray(tensors[_key(p)]).reshape(leaf.shape).astype(leaf.dtype)
        for p, leaf in leaves_with_path
    ]
    return jax.tree_util.tree_unflatten(treedef, leaves)


__all__ = ["save_model", "load_model"]
