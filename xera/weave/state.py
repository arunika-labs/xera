

"""
State management base class for training and optimization states.

This module provides the State base class, which is similar to Module but
designed for managing training states, optimizer states, and other stateful
components that don't require random key management.
"""

from __future__ import annotations
import dataclasses
import jax
import jax.numpy as jnp


class State:
    """
    Base class for stateful components in the training framework.
    
    State is similar to Module but designed for components that don't need
    random key management (like optimizer states, training loops, etc.).
    It provides automatic JAX pytree registration and a setup hook for
    initialization logic.
    
    Subclasses should define their state fields as class attributes and
    can override the setup method for initialization logic.
    
    Example:
        >>> class MyState(State):
        ...     step: int = 0
        ...     total_loss: float = 0.0
        ...     
        ...     def setup(self):
        ...         print("State initialized")
        ...
        >>> state = MyState(step=100, total_loss=5.5)
    """

    def __init_subclass__(cls, **kwargs):
        """
        Automatically register subclasses as dataclasses and JAX pytrees.
        
        This method is called when a subclass is created and automatically:
        - Converts the class to a dataclass with specific configurations
        - Registers the class with JAX's pytree utilities for JIT compilation
        """
        super().__init_subclass__(**kwargs)
        cls = dataclasses.dataclass(cls, eq=False, repr=False, init=False)
        jax.tree_util.register_pytree_node(
            cls,
            cls._tree_flatten,
            cls._tree_unflatten,
        )

    def __new__(cls, *args, **kwargs):
        """Create a new instance of the state."""
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        """
        Initialize the state with field values.
        
        Args:
            *args: Positional arguments corresponding to state fields.
            **kwargs: Keyword arguments for state fields.
        """
        field_names = [f.name for f in dataclasses.fields(self)]
        positional = dict(zip(field_names, args))
        for name, val in {**positional, **kwargs}.items():
            object.__setattr__(self, name, val)
        self.setup()

    def setup(self):
        """
        Initialize state components.
        
        This method is called during __init__ and should be overridden by
        subclasses to perform any initialization logic.
        
        Example:
            >>> def setup(self):
            ...     self.step = 0
            ...     self.best_loss = float('inf')
        """
        pass

    
    def _tree_flatten(self):
        """
        Flatten the state for JAX pytree operations.
        
        This method separates the state's attributes into dynamic values
        (arrays, states, modules) that should be part of the pytree structure
        and static values (hyperparameters, configuration) that should remain
        constant during transformations.
        
        Returns:
            A tuple (dynamic_vals, aux_data) where:
                - dynamic_vals: Tuple of dynamic attribute values
                - aux_data: Auxiliary data for reconstruction (names, static values)
        """
        dynamic_names, dynamic_vals = [], []
        static_names, static_vals = [], []
        for name, val in self.__dict__.items():
            if isinstance(val, (jnp.ndarray, State)) or val is None:
                dynamic_names.append(name)
                dynamic_vals.append(val)
            else:
                # Import Module here to avoid circular imports
                from ..core import Module as _Module
                if isinstance(val, _Module):
                    dynamic_names.append(name)
                    dynamic_vals.append(val)
                else:
                    static_names.append(name)
                    static_vals.append(val)
        aux_data = (tuple(dynamic_names), tuple(static_names), tuple(static_vals))
        return tuple(dynamic_vals), aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a state from flattened representation.
        
        This is the inverse operation of _tree_flatten, used by JAX to
        reconstruct states after transformations like JIT or grad.
        
        Args:
            aux_data: Auxiliary data containing attribute names and static values
            children: Dynamic attribute values from the flattened representation
        
        Returns:
            A reconstructed state instance.
        """
        dynamic_names, static_names, static_vals = aux_data
        obj = object.__new__(cls)
        for name, val in zip(dynamic_names, children):
            object.__setattr__(obj, name, val)
        for name, val in zip(static_names, static_vals):
            object.__setattr__(obj, name, val)
        return obj

    def __repr__(self):
        """
        Return a string representation of the state.
        
        Returns:
            A string showing the state class name and its field values.
        """
        field_names = [f.name for f in dataclasses.fields(self)]
        parts = ", ".join(f"{n}={getattr(self, n)!r}" for n in field_names)
        return f"{type(self).__name__}({parts})"