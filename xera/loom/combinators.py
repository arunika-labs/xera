

"""
Combinator modules for building complex neural network architectures.

This module provides utility modules for combining layers in common patterns:
sequential composition, residual connections, and custom function wrapping.
"""

from __future__ import annotations
from typing import Any
from ..core import Module
from .stochastic import Dropout


class Sequential(Module):
    """
    Sequential layer composition.
    
    Applies a sequence of layers in order, passing the output of each layer
    as input to the next. Dropout layers receive keyword arguments (like
    key and deterministic) while other layers do not.
    
    Attributes:
        layers: A list of Module instances to apply sequentially.
    
    Example:
        >>> model = Sequential([
        ...     Dense(784, 256),
        ...     Dense(256, 128),
        ...     Dense(128, 10)
        ... ])
        >>> output = model(input_tensor)
    """
    
    layers: list

    def setup(self):
        """No setup needed for Sequential."""
        pass

    def __call__(self, x, **kwargs):
        """
        Apply layers sequentially to the input.
        
        Args:
            x: Input tensor.
            **kwargs: Keyword arguments passed to Dropout layers.
        
        Returns:
            Output after applying all layers sequentially.
        """
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer(x, **kwargs)
            else:
                x = layer(x)
        return x


class Residual(Module):
    """
    Residual (skip) connection wrapper.
    
    Adds the input to the output of the inner layer, implementing the
    residual connection pattern from ResNet. This helps with gradient
    flow in very deep networks.
    
    Attributes:
        inner: The module to wrap with a residual connection.
    
    Example:
        >>> block = Residual(Dense(256, 256))
        >>> output = block(input_tensor)  # Applies Dense(x) + x
    """

    inner: Any

    def setup(self):
        """No setup needed for Residual."""
        pass

    def __call__(self, x, **kwargs):
        """
        Apply the inner layer and add the input (residual connection).
        
        Args:
            x: Input tensor.
            **kwargs: Keyword arguments passed to the inner layer.
        
        Returns:
            Output of inner layer plus the original input.
        """
        return x + self.inner(x, **kwargs)


class Lambda(Module):
    """
    Wrapper for arbitrary functions.
    
    Allows custom functions to be used as modules in the framework.
    Useful for custom transformations, activations, or other operations
    that don't require parameters.
    
    Attributes:
        fn: The function to apply to the input.
    
    Example:
        >>> custom_relu = Lambda(lambda x: jnp.maximum(0, x))
        >>> output = custom_relu(input_tensor)
    """

    fn: Any

    def setup(self):
        """No setup needed for Lambda."""
        pass

    def __call__(self, x, **kwargs):
        """
        Apply the wrapped function to the input.
        
        Args:
            x: Input tensor.
            **kwargs: Keyword arguments (ignored by default function).
        
        Returns:
            Output of the wrapped function.
        """
        return self.fn(x)


__all__ = ["Sequential", "Residual", "Lambda"]
