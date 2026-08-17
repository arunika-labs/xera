

"""
Model serialization utilities for saving and loading neural network models.

This module provides functions to save and load model parameters using the
safetensors format, which is a safe and efficient format for storing tensors.
"""

from __future__ import annotations
import jax
import numpy as np
from safetensors.numpy import save_file, load_file


def _key(path):
    """
    Convert a JAX tree path to a string key.
    
    Args:
        path: A JAX tree path object.
    
    Returns:
        A string representation of the path with leading dots removed.
    """
    return jax.tree_util.keystr(path).lstrip(".")


def save_model(module, path):
    """
    Save a model's parameters to a safetensors file.
    
    This function flattens the model's parameter tree and saves all
    parameters to a safetensors file, which is a safe and efficient
    format for storing tensors.
    
    Args:
        module: The model module to save (should be a JAX pytree).
        path: The file path where the model should be saved.
    
    Example:
        >>> save_model(my_model, "model.safetensors")
    """
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(module)
    tensors = {_key(p): np.asarray(leaf) for p, leaf in leaves_with_path}
    save_file(tensors, path)


def load_model(template, path):
    """
    Load a model's parameters from a safetensors file.
    
    This function loads parameters from a safetensors file and reconstructs
    the model using a template to determine the structure. The template
    should have the same architecture as the saved model.
    
    Args:
        template: A model instance with the same architecture as the saved model.
        path: The file path to load the model from.
    
    Returns:
        A model instance with loaded parameters.
    
    Example:
        >>> template = MyModel()  # Create empty model with same architecture
        >>> loaded_model = load_model(template, "model.safetensors")
    """
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(template)
    tensors = load_file(path)
    leaves = [
        jax.numpy.asarray(tensors[_key(p)]).reshape(leaf.shape).astype(leaf.dtype)
        for p, leaf in leaves_with_path
    ]
    return jax.tree_util.tree_unflatten(treedef, leaves)


__all__ = ["save_model", "load_model"]
