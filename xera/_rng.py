"""
Small internal RNG utility shared by `loom.Module` and `weave.Struct`.

This is intentionally not a "core" abstractions module -- it holds
exactly one thing (`RNGPool`) that both `Module` and `Struct` need for
their `self.rng()` helper, with no dependency in either direction
between `loom` and `weave`.
"""

from __future__ import annotations
import jax


class RNGPool:
    """
    A pool for managing JAX random number generation keys.

    This class provides a convenient way to manage random keys for
    stochastic operations. It maintains an internal key that gets split
    each time a new random key is requested, ensuring reproducible and
    independent random numbers.

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


__all__ = ["RNGPool"]
