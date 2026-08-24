"""
Core module providing fundamental abstractions for the xera framework.

This module defines the base classes and utilities used throughout the framework:
- RNGPool: Manages random number generation for stochastic operations
- Buffer: A wrapper for values that should be treated as leaf nodes in JAX trees
- Module: Base class for all neural network components
- Struct: Base class for training-side components (datasets, optimizers,
  train steps, and other non-parameter pytrees)
- param: Helper function for parameter initialization
"""

from __future__ import annotations
import dataclasses
import jax
import jax.numpy as jnp


class RNGPool:
    """
    A pool for managing JAX random number generation keys.
    
    This class provides a convenient way to manage random keys for stochastic
    operations in neural networks. It maintains an internal key that gets
    split each time a new random key is requested, ensuring reproducible
    and independent random numbers.
    
    Attributes:
        _key: The internal JAX PRNG key.
    
    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> pool = RNGPool(key)
        >>> subkey1 = pool.next()
        >>> subkeys = pool.split(3)  # Get 3 independent keys
    """
    
    __slots__ = ("_key",)

    def __init__(self, key):
        """
        Initialize the RNG pool with a PRNG key.
        
        Args:
            key: A JAX PRNG key (typically from jax.random.PRNGKey).
        """
        self._key = key

    def next(self):
        """
        Get a single new random key from the pool.
        
        Returns:
            A new JAX PRNG key that can be used for random operations.
            The internal key is updated to ensure future calls return
            independent keys.
        """
        self._key, sub = jax.random.split(self._key)
        return sub

    def split(self, n):
        """
        Get multiple new random keys from the pool.
        
        Args:
            n: The number of random keys to generate.
        
        Returns:
            A list of n independent JAX PRNG keys. The internal key is
            updated to ensure future calls return independent keys.
        """
        self._key, *subs = jax.random.split(self._key, n + 1)
        return subs


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


class Struct:
    """
    Base class for training-state components in the xera framework.

    Struct is the training-side counterpart to `Module`: same mechanics
    (dataclass fields, keyed JAX pytree registration, an optional
    `setup()` hook, optional `self.rng()`), but meant for things that
    drive or observe training rather than for differentiable model
    parameters. Use it for datasets, parallel/sharding wrappers,
    optimizers, train steps, and the top-level training driver -- all
    as plain `Struct` subclasses, composed by holding each other (and
    `Module` instances) as fields.

    Example:
        >>> class Datasets(Struct):
        ...     x: jnp.ndarray = None
        ...     y: jnp.ndarray = None
        ...
        ...     def augment(self):
        ...         noise = jax.random.normal(self.rng(), self.x.shape)
        ...         return self.x + 0.01 * noise
        ...
        >>> class Trainer(Struct):
        ...     model: Module = None
        ...     data: Datasets = None
        ...     optimizer: "Optimizer" = None
        ...
        ...     def run(self):
        ...         ...  # user-defined training loop
        ...
        >>> trainer = Trainer(model=my_model, data=Datasets(x=xs, y=ys, key=k),
        ...                    optimizer=Adam(lr=1e-3))
        >>> trainer.run()
    """

    def __init_subclass__(cls, **kwargs):
        """
        Automatically register subclasses as dataclasses and JAX pytrees.

        Mirrors `Module.__init_subclass__`: converts the subclass into a
        dataclass and registers it with JAX's keyed pytree utilities so
        it can flow through `jit`/`grad`/`scan` with readable attribute
        paths in error messages.
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
        """Create a new instance of the struct."""
        return super().__new__(cls)

    def __init__(self, *args, key=None, **kwargs):
        """
        Initialize the struct with field values and an optional random key.

        After field assignment, calls `self.setup()`, and then -- if the
        subclass defines its own `run()` (i.e. it's not just the no-op
        base `Struct.run`) -- calls `self.run()` too. This is what lets a
        `Trainer(Struct)` with a `run()` method start training simply by
        being instantiated: `Trainer(key=k, ...)`.

        Args:
            *args: Positional arguments corresponding to struct fields.
            key: Optional JAX PRNG key. If provided, enables `self.rng()`
                during `setup()`/`run()` (e.g. for dataset augmentation,
                or as the training loop's root key).
            **kwargs: Keyword arguments for struct fields.

        Raises:
            RuntimeError: If `self.rng()` is called during `setup()`/`run()`
                but no `key=` was provided.
        """
        field_names = [f.name for f in dataclasses.fields(self)]
        positional = dict(zip(field_names, args))
        for name, val in {**positional, **kwargs}.items():
            object.__setattr__(self, name, val)

        if key is not None:
            object.__setattr__(self, "_rng_pool", RNGPool(key))
        self.setup()

        if type(self).run is not Struct.run:
            self.run()

    def setup(self):
        """
        Initialize struct fields and nested components.

        Called during `__init__`; override in subclasses to perform any
        initialization logic. Use `self.rng()` for randomness that needs
        an explicitly-provided key.

        Example:
            >>> def setup(self):
            ...     self.step = 0
            ...     self.best_loss = float('inf')
        """
        pass

    def run(self):
        """
        Entry point for structs that represent a runnable process (e.g. a
        `Trainer`). No-op by default.

        If a subclass overrides `run()`, `__init__` calls it automatically
        right after `setup()`, so instantiating the struct is enough to
        start it -- no separate `.run()` call needed:

            >>> class Trainer(Struct):
            ...     def setup(self):
            ...         self.model = MyModel(key=self.rng())
            ...         self.optimizer = Adam(lr=1e-3)
            ...     def run(self):
            ...         ...  # define body_fn, call weave.loop(...), etc.
            ...
            >>> trainer = Trainer(key=jax.random.PRNGKey(0))  # setup() then run()

        Structs that aren't runnable processes (datasets, loop configs,
        wrapped optimizer state, ...) simply don't override `run`, and
        this base no-op is skipped.
        """
        pass

    def rng(self, n=None):
        """
        Get random keys from the struct's RNG pool.

        Args:
            n: Optional number of random keys to generate. If None,
                returns a single key. If specified, returns n keys.

        Returns:
            A single JAX PRNG key if n is None, or a list of n keys.

        Raises:
            RuntimeError: If the struct was created without a `key=`
                parameter. This is intentional: silently falling back to
                a default key would hide non-determinism, which conflicts
                with JAX's explicit, functional state model.
        """
        pool = getattr(self, "_rng_pool", None)
        if pool is None:
            raise RuntimeError(
                "self.rng() dipanggil tapi Struct ini dibuat tanpa `key=`."
            )
        return pool.split(n) if n is not None else pool.next()

    @staticmethod
    def _is_dynamic(val):
        """
        Decide whether a field value belongs in the dynamic pytree part.

        Dynamic: JAX arrays, `Module`/`Struct` instances (and `None`),
        plus `list`/`dict` values whose elements are all `Module` and/or
        `Struct` instances (mirrors the list-of-Module exception in
        `Module._tree_flatten`, extended to `dict` per design decision).
        Everything else (plain config, callables, hyperparameters) is
        static.
        """
        if isinstance(val, (jnp.ndarray, Module, Struct)) or val is None:
            return True
        if isinstance(val, list) and val and all(
            isinstance(v, (Module, Struct)) for v in val
        ):
            return True
        if isinstance(val, dict) and val and all(
            isinstance(v, (Module, Struct)) for v in val.values()
        ):
            return True
        return False

    def _tree_flatten(self):
        """
        Flatten the struct for JAX pytree operations.

        Separates attributes into dynamic values (arrays, Structs,
        Modules, and list/dict thereof) that participate in the pytree,
        and static values (config, hyperparameters, callables) that stay
        constant across transformations.

        Returns:
            A tuple (dynamic_vals, aux_data) where:
                - dynamic_vals: Tuple of dynamic attribute values
                - aux_data: Auxiliary data for reconstruction (names, static values)
        """
        dynamic_names, dynamic_vals = [], []
        static_names, static_vals = [], []
        for name, val in self.__dict__.items():
            if self._is_dynamic(val):
                dynamic_names.append(name)
                dynamic_vals.append(val)
            else:
                static_names.append(name)
                static_vals.append(val)
        aux_data = (tuple(dynamic_names), tuple(static_names), tuple(static_vals))
        return tuple(dynamic_vals), aux_data

    def _tree_flatten_with_keys(self):
        """
        Flatten the struct with attribute keys for JAX pytree operations.

        Same as `_tree_flatten`, but pairs each dynamic value with a
        `GetAttrKey` so JAX transformations report readable attribute
        paths (e.g. `trainer.data.x`) in error messages.

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
        Reconstruct a struct from flattened representation.

        Inverse of `_tree_flatten`, used by JAX to rebuild the struct
        after transformations like `jit`, `grad`, or `scan`.

        Args:
            aux_data: Auxiliary data containing attribute names and static values
            children: Dynamic attribute values from the flattened representation

        Returns:
            A reconstructed struct instance.
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
        Return a string representation of the struct.

        Returns:
            A string showing the struct class name and its field values.
        """
        field_names = [f.name for f in dataclasses.fields(self)]
        parts = ", ".join(f"{n}={getattr(self, n)!r}" for n in field_names)
        return f"{type(self).__name__}({parts})"

    def save_struct(self, model, optimizer, metadata, path):
        """
        Save a model, an optimizer (state), and metadata together as one
        `.sxera` checkpoint file.

        This is a thin wrapper around `xera.serialize.sxera.save_struct` --
        see that function for the on-disk format. Typically called from
        inside `run()` (e.g. via a `Callback.io` checkpoint hook):

            >>> self.save_struct(
            ...     self.model, self.optimizer,
            ...     metadata={"step": step, "key": self.rng()},
            ...     path="ckpt.sxera",
            ... )

        Args:
            model: The model (`Module` instance) to save.
            optimizer: The optimizer (state) to save.
            metadata: A dict (or `Struct`) of everything else needed to
                resume training -- step counters, RNG keys, config, and
                the like.
            path: Destination file path (conventionally ending in `.sxera`).
        """
        from .serialize.sxera import save_struct as _save_struct

        _save_struct(model, optimizer, metadata, path)


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
            in xera.initializers (e.g., normal(), xavier_normal(), etc.).
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
