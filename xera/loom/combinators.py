"""
Combinator modules for building complex neural network architectures.

This module provides utility modules for combining layers in common patterns:
sequential composition, residual connections, and custom function wrapping.
"""

from __future__ import annotations
import inspect
from typing import Any
from .module import Module


def _accepted_kwargs(layer, kwargs):
    """
    Filter kwargs down to the ones a layer's __call__ actually accepts.

    Inspects the layer's __call__ signature so kwargs like `key`,
    `deterministic`, or `mask` are only forwarded to layers that declare
    them, instead of hardcoding a fixed list of "stochastic" layer types.
    Layers that declare **kwargs in their signature receive everything.

    Args:
        layer: The Module instance being called.
        kwargs: The full kwargs dict available to forward.

    Returns:
        A dict containing only the kwargs accepted by layer.__call__.
    """
    if not kwargs:
        return {}
    try:
        sig = inspect.signature(layer.__call__)
    except (TypeError, ValueError):
        return {}
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _call_layer(layer, x, **kwargs):
    """
    Call a layer with only the kwargs it accepts, and normalize its output.

    Some layers (e.g. BatchNorm, GroupNormWithRunningStats) are stateful:
    they return a (output, new_layer) tuple so updated running statistics
    can be threaded back into the model. This helper detects that pattern
    and returns both the output and the (possibly updated) layer, so
    callers don't need to special-case specific layer classes.

    Args:
        layer: The Module instance to call.
        x: Input tensor.
        **kwargs: Candidate keyword arguments, filtered per-layer.

    Returns:
        A tuple (output, new_layer). new_layer is the original layer
        unchanged unless the layer returned updated state.
    """
    call_kwargs = _accepted_kwargs(layer, kwargs)
    result = layer(x, **call_kwargs)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Module):
        return result[0], result[1]
    return result, layer


class Sequential(Module):
    """
    Sequential layer composition.
    
    Applies a sequence of layers in order, passing the output of each layer
    as input to the next. Keyword arguments (like `key` and `deterministic`)
    are forwarded to each layer only if that layer's `__call__` declares
    them, so stochastic layers (Dropout) and stateful layers (BatchNorm)
    both receive what they need without special-casing specific types.
    
    Attributes:
        layers: A list of Module instances to apply sequentially.
    
    Example:
        >>> model = Sequential([
        ...     Dense(784, 256),
        ...     Dense(256, 128),
        ...     Dense(128, 10)
        ... ])
        >>> output = model(input_tensor)

        >>> # With stochastic and stateful layers, e.g. during eval:
        >>> model = Sequential([Dense(64, 64), BatchNorm(dim=64), Dropout(rate=0.1)])
        >>> output, new_model = model(input_tensor, deterministic=True)
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
            **kwargs: Keyword arguments (e.g. `key`, `deterministic`, `mask`)
                forwarded to each layer that declares them.
        
        Returns:
            If any layer in the sequence is stateful (returns updated
            state, e.g. BatchNorm), returns a tuple (output, new_self)
            where new_self is a Sequential with updated layers. Otherwise
            returns just the output tensor.
        """
        new_layers = list(self.layers)
        stateful = False
        for i, layer in enumerate(self.layers):
            x, new_layer = _call_layer(layer, x, **kwargs)
            if new_layer is not layer:
                stateful = True
            new_layers[i] = new_layer
        if stateful:
            new_self = object.__new__(type(self))
            new_self.__dict__.update(self.__dict__)
            new_self.layers = new_layers
            return x, new_self
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
            **kwargs: Keyword arguments forwarded to the inner layer if
                its `__call__` declares them.
        
        Returns:
            If the inner layer is stateful (e.g. BatchNorm), returns a
            tuple (output, new_self) with the updated inner layer.
            Otherwise returns just the output tensor (inner(x) + x).
        """
        out, new_inner = _call_layer(self.inner, x, **kwargs)
        result = x + out
        if new_inner is not self.inner:
            new_self = object.__new__(type(self))
            new_self.__dict__.update(self.__dict__)
            new_self.inner = new_inner
            return result, new_self
        return result


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
