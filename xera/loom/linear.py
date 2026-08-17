

"""
Linear (dense) layers for fully-connected neural network operations.

This module provides the standard dense/fully-connected layer that applies
a linear transformation to input data, optionally followed by a bias addition.
"""

from __future__ import annotations
import jax.numpy as jnp
from ..core import Module, param
from .. import initializers


class Dense(Module):
    """
    Standard dense (fully-connected) layer.
    
    Applies a linear transformation to the input: y = x @ W + b
    where W is the weight matrix and b is the optional bias vector.
    This is the fundamental building block for most neural networks.
    
    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        use_bias: Whether to add a bias term (default: True).
    
    Example:
        >>> layer = Dense(in_features=128, out_features=64)
        >>> output = layer(input_tensor)  # shape: (..., 64)
    """
    
    in_features: int
    out_features: int
    use_bias: bool = True

    def setup(self):
        """Initialize weight matrix and optional bias vector."""
        self.weight = param(
            self.rng(), initializers.lecun_normal(),
            (self.in_features, self.out_features),
        )
        self.bias = (
            param(self.rng(), initializers.zeros(), (self.out_features,))
            if self.use_bias else None
        )

    def __call__(self, x):
        """
        Apply the linear transformation to the input.
        
        Args:
            x: Input tensor of shape (..., in_features).
        
        Returns:
            Output tensor of shape (..., out_features).
        """
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


__all__ = ["Dense"]