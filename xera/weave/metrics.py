

"""
Metrics logging system for tracking training progress.

This module provides a flexible metrics logging system that allows custom
emission functions to be registered for different metric names, with a
default fallback to console output.
"""

from __future__ import annotations
import jax


def _default_emit(name, step, value):
    """
    Default emission function that prints metrics to console.
    
    Args:
        name: The name of the metric.
        step: The training step number (can be None).
        value: The metric value.
    """
    if step is None:
        print(f"{name} = {float(value):.6f}")
    else:
        print(f"[step {int(step)}] {name} = {float(value):.6f}")


class Metrics:
    """
    A flexible metrics logging system with custom emission functions.
    
    This class allows registering custom emission functions for specific
    metric names, enabling integration with various logging backends
    (TensorBoard, Weights & Biases, custom dashboards, etc.).
    
    Attributes:
        _registry: Dictionary mapping metric names to emission functions.
    
    Example:
        >>> # Register a custom logger
        >>> def custom_logger(step, value):
        ...     print(f"Custom: step={step}, value={value}")
        >>> Metrics.register("custom_metric", custom_logger)
        >>> 
        >>> # Log metrics during training
        >>> Metrics.log(step=100, loss=0.5, accuracy=0.9)
    """

    _registry = {}

    @classmethod
    def register(cls, name, fn=None):
        """
        Register a custom emission function for a metric name.
        
        Can be used as a decorator or called directly.
        
        Args:
            name: The metric name to register.
            fn: The emission function. If None, returns a decorator.
        
        Returns:
            The emission function if fn is provided, otherwise a decorator.
        
        Example:
            >>> @Metrics.register("my_metric")
            ... def my_logger(step, value):
            ...     print(f"My metric: {value}")
        """
        def deco(f):
            cls._registry[name] = f
            return f
        if fn is not None:
            cls._registry[name] = fn
            return fn
        return deco

    @classmethod
    def unregister(cls, name):
        """
        Unregister an emission function for a metric name.
        
        Args:
            name: The metric name to unregister.
        """
        cls._registry.pop(name, None)

    @classmethod
    def log(cls, step=None, **values):
        """
        Log metric values using registered emission functions.
        
        For each metric, uses the registered emission function if available,
        otherwise falls back to the default console output.
        
        Args:
            step: Optional training step number.
            **values: Keyword arguments of metric names and values.
        
        Example:
            >>> Metrics.log(step=100, loss=0.5, accuracy=0.9, custom_metric=0.3)
        """
        def _emit(step, values):
            for name, value in values.items():
                fn = cls._registry.get(name, None)
                if fn is not None:
                    fn(step, value)
                else:
                    _default_emit(name, step, value)
        jax.debug.callback(_emit, step, values)


__all__ = ["Metrics"]
