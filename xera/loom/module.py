"""
Base abstractions for `loom` neural network layers.

Defines:
- Buffer: A wrapper for values that should be treated as leaf nodes in JAX trees
- Module: Base class for all neural network components
- param: Helper function for parameter initialization

Moved out of the old shared `xera.core` module -- these are used
exclusively by `loom` (layers), so they live here now instead of in a
cross-cutting "core" module.
"""

from __future__ import annotations
import dataclasses
import jax
import jax.numpy as jnp
from .._rng import RNGPool


class Buffer:
    """
    A wrapper for values that should be treated as leaf nodes in JAX trees.

    Buffer is used to wrap values that need to be part of the JAX pytree
    structure but should not be traversed further. This is particularly
    useful for running statistics in normalization layers and other
    stateful components.

    Attributes:
        value: The wrapped value.

    Example:
        >>> buffer = Buffer(jnp.zeros((10,)))
        >>> # The buffer will be treated as a leaf in JAX tree operations
    """

    __slots__ = ("value",)

    def __init__(self, value):
        """
        Initialize a buffer with a value.

        Args:
            value: The value to wrap in the buffer.
        """
        self.value = value

    def __repr__(self):
        """Return a string representation of the buffer."""
        return f"Buffer({self.value!r})"


jax.tree_util.register_pytree_with_keys(
    Buffer,
    lambda s: ([(jax.tree_util.GetAttrKey("value"), s.value)], None),
    lambda aux, children: Buffer(children[0]),
)


class Module:
    """
    Base class for all neural network modules in the xera framework.

    Module provides the foundation for building neural network components with:
    - Automatic JAX pytree registration for JIT compilation
    - Parameter initialization with random key management
    - Hierarchical module composition
    - Setup hooks for initialization logic

    Subclasses should define their parameters as class attributes and
    implement the __call__ method for forward computation.

    Example:
        >>> class MyLinear(Module):
        ...     in_features: int
        ...     out_features: int
        ...
        ...     def setup(self):
        ...         self.weight = param(self.rng(), glorot_normal(),
        ...                           (self.in_features, self.out_features))
        ...
        ...     def __call__(self, x):
        ...         return x @ self.weight
        ...
        >>> layer = MyLinear(10, 20, key=jax.random.PRNGKey(0))
        >>> output = layer(jnp.ones((5, 10)))
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
        jax.tree_util.register_pytree_with_keys(
            cls,
            cls._tree_flatten_with_keys,
            cls._tree_unflatten,
            cls._tree_flatten,
        )

    def __new__(cls, *args, **kwargs):
        """Create a new instance of the module."""
        return super().__new__(cls)

    def __init__(self, *args, key=None, **kwargs):
        """
        Initialize the module with parameters and optional random key.

        Args:
            *args: Positional arguments corresponding to module fields.
            key: Optional JAX PRNG key for parameter initialization. If provided,
                enables the use of self.rng() during setup.
            **kwargs: Keyword arguments for module fields.

        Raises:
            RuntimeError: If self.rng() is called during setup but no key was provided.
        """
        field_names = [f.name for f in dataclasses.fields(self)]
        positional = dict(zip(field_names, args))
        for name, val in {**positional, **kwargs}.items():
            object.__setattr__(self, name, val)

        if key is not None:
            object.__setattr__(self, "_rng_pool", RNGPool(key))
        self.setup()
        if key is not None:
            object.__delattr__(self, "_rng_pool")

    def setup(self):
        """
        Initialize module parameters and submodules.

        This method is called during __init__ and should be overridden by
        subclasses to initialize parameters and create submodules. Use
        self.rng() to get random keys for parameter initialization.

        Example:
            >>> def setup(self):
            ...     self.weight = param(self.rng(), normal(), (10, 20))
            ...     self.bias = param(self.rng(), zeros(), (20,))
        """
        pass

    def rng(self, n=None):
        """
        Get random keys from the module's RNG pool.

        Args:
            n: Optional number of random keys to generate. If None, returns
                a single key. If specified, returns n keys.

        Returns:
            A single JAX PRNG key if n is None, or a list of n keys.

        Raises:
            RuntimeError: If the module was created without a key parameter.
        """
        pool = getattr(self, "_rng_pool", None)
        if pool is None:
            raise RuntimeError(
                "self.rng() dipanggil tapi Module ini dibuat tanpa `key=`."
            )
        return pool.split(n) if n is not None else pool.next()

    def __call__(self, *args, **kwargs):
        """
        Forward pass of the module.

        This method must be implemented by subclasses to define the
        forward computation of the module.

        Args:
            *args: Input arguments to the module.
            **kwargs: Keyword arguments to the module.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    def _tree_flatten(self):
        """
        Flatten the module for JAX pytree operations.

        This method separates the module's attributes into dynamic values
        (arrays, modules, buffers) that should be part of the pytree structure
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
            if isinstance(val, (jnp.ndarray, Module, Buffer)) or val is None:
                dynamic_names.append(name)
                dynamic_vals.append(val)
            elif isinstance(val, list) and val and all(isinstance(v, Module) for v in val):
                dynamic_names.append(name)
                dynamic_vals.append(val)
            else:
                static_names.append(name)
                static_vals.append(val)
        aux_data = (tuple(dynamic_names), tuple(static_names), tuple(static_vals))
        return tuple(dynamic_vals), aux_data

    def _tree_flatten_with_keys(self):
        """
        Flatten the module with attribute keys for JAX pytree operations.

        Similar to _tree_flatten, but includes attribute keys for better
        debugging and error messages in JAX transformations.

        Returns:
            A tuple (keyed_children, aux_data) where:
                - keyed_children: List of (key, value) pairs for dynamic attributes
                - aux_data: Auxiliary data for reconstruction
        """
        dynamic_vals, aux_data = self._tree_flatten()
        dynamic_names = aux_data[0]
        keyed = [
            (jax.tree_util.GetAttrKey(name), val)
            for name, val in zip(dynamic_names, dynamic_vals)
        ]
        return keyed, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a module from flattened representation.

        This is the inverse operation of _tree_flatten, used by JAX to
        reconstruct modules after transformations like JIT or grad.

        Args:
            aux_data: Auxiliary data containing attribute names and static values
            children: Dynamic attribute values from the flattened representation

        Returns:
            A reconstructed module instance.
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
        Return a string representation of the module.

        Returns:
            A string showing the module class name and its field values.
        """
        field_names = [f.name for f in dataclasses.fields(self)]
        parts = ", ".join(f"{n}={getattr(self, n)!r}" for n in field_names)
        return f"{type(self).__name__}({parts})"


def param(key, init_fn, shape, dtype=jnp.float32):
    """
    Initialize a parameter using the given initialization function.

    This is a convenience function for creating parameters in modules.
    It applies the initialization function to generate parameter values
    with the specified shape and dtype.

    Args:
        key: A JAX PRNG key for random initialization.
        init_fn: An initialization function that takes (key, shape, dtype)
            and returns an initialized array. Common functions are defined
            in xera.loom.initializers (e.g., normal(), xavier_normal(), etc.).
        shape: The shape of the parameter to create.
        dtype: The data type of the parameter (default: jnp.float32).

    Returns:
        An initialized parameter array with the specified shape and dtype.

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> weight = param(key, initializers.xavier_normal(), (10, 20))
        >>> bias = param(key, initializers.zeros(), (20,))
    """
    return init_fn(key, shape, dtype)


__all__ = ["Module", "Buffer", "param"]
