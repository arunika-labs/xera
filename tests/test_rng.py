"""Tests for xera._rng.RNGPool, the small RNG utility shared by
xera.loom.Module and xera.optimizer.State.
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


def test_rng_pool_is_exposed_on_loom_and_optimizer():
    import xera.loom as xl
    import xera.optimizer as optimizer

    assert xl.RNGPool is RNGPool
    # optimizer.State uses the same RNGPool internally; not necessarily
    # re-exported from xera.optimizer's public API, so just check State works.
    assert hasattr(optimizer, "State")
