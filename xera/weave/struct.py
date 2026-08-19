"""
Struct base class for training-state components.

This module provides the Struct base class: a single, generic base for
every non-parameter, training-side component in `xera.weave` (datasets,
callbacks, parallel wrappers, train steps, validation loops, and the
top-level training driver itself). Mechanically it is a near-exact copy
of `xera.core.Module` -- same dataclass-on-subclass registration, same
keyed-pytree flatten/unflatten, same optional `self.rng()` -- but scoped
to training state rather than model parameters. A `Train` is not a
special built-in class: it is just a `Struct` whose fields happen to be
other `Struct`/`Module` instances (e.g. a dataset, a set of callbacks, an
optimizer-driving train step), with a `run()` method the user writes
themselves.

`Struct.rng()` follows the same explicit, no-hidden-state rule as
`Module.rng()`: if a `Struct` is constructed without `key=`, calling
`self.rng()` during `setup()` raises `RuntimeError` rather than silently
falling back to a default key. Implicitly injecting `PRNGKey(0)` would
violate JAX's functional/explicit-state discipline, so callers that need
randomness (e.g. a `Datasets` struct doing augmentation) must pass
`key=` explicitly, exactly like a `Module`.
"""

from __future__ import annotations
import dataclasses
import jax
import jax.numpy as jnp
from ..core import Module, RNGPool


class Struct:
    """
    Base class for training-state components in the xera framework.

    Struct is the training-side counterpart to `Module`: same mechanics
    (dataclass fields, keyed JAX pytree registration, an optional
    `setup()` hook, optional `self.rng()`), but meant for things that
    drive or observe training rather than for differentiable model
    parameters. Use it for datasets, callbacks, parallel/sharding
    wrappers, train steps, validation loops, and the top-level training
    driver -- all as plain `Struct` subclasses, composed by holding each
    other as fields.

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

        Args:
            *args: Positional arguments corresponding to struct fields.
            key: Optional JAX PRNG key. If provided, enables `self.rng()`
                during `setup()` (e.g. for dataset augmentation).
            **kwargs: Keyword arguments for struct fields.

        Raises:
            RuntimeError: If `self.rng()` is called during `setup()` but
                no `key=` was provided.
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


__all__ = ["Struct"]
