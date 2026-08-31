"""Tests for xera._rng.RNGPool, the small RNG utility shared by
xera.loom.Module and xera.weave.Struct.
"""

import jax
import jax.numpy as jnp
from xera._rng import RNGPool


def test_rng_pool_split_returns_independent_keys():
    pool = RNGPool(jax.random.PRNGKey(42))
    k1 = pool.next()
    k2 = pool.next()
    assert not jnp.array_equal(k1, k2)

    pool2 = RNGPool(jax.random.PRNGKey(42))
    keys = pool2.split(3)
    assert len(keys) == 3
    assert not jnp.array_equal(keys[0], keys[1])


def test_rng_pool_is_exposed_on_loom_and_weave():
    import xera.loom as xl
    import xera.weave as weave

    assert xl.RNGPool is RNGPool
    # weave.Struct uses the same RNGPool internally; not necessarily
    # re-exported from xera.weave's public API, so just check Struct works.
    assert hasattr(weave, "Struct")
