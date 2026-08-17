"""Tests for xera.core: Module base class, pytree registration, RNG pooling."""

import jax
import jax.numpy as jnp
import xera
import xera.loom as loom


def test_forward_and_auto_rng_split():
    block = loom.TransformerBlock(dim=32, num_heads=4, mlp_hidden_dim=64, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 32))
    out = block(x, deterministic=True)
    assert out.shape == (2, 8, 32)
    assert not jax.numpy.allclose(block.attn.q_proj.weight, block.attn.k_proj.weight)


def test_jit_and_grad():
    block = loom.TransformerBlock(dim=32, num_heads=4, mlp_hidden_dim=64, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 32))

    fwd = jax.jit(lambda b, x: b(x, deterministic=True))
    out = fwd(block, x)
    assert out.shape == (2, 8, 32)

    grads = jax.grad(lambda b, x: jnp.sum(fwd(b, x) ** 2))(block, x)
    assert grads.attn.q_proj.weight.shape == (32, 32)


def test_params_pytree_leaves():
    # params_dict()/state_dict() were removed (dead code, duplicated what
    # native pytree flattening already does) -- this is the replacement:
    # inspect params via jax.tree_util directly, same route `serialize`
    # and `grad` already use.
    dense = loom.Dense(4, 8, key=jax.random.PRNGKey(0))
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(dense)
    names = {jax.tree_util.keystr(p) for p, _ in leaves_with_path}
    assert names == {".weight", ".bias"}
    shapes = {jax.tree_util.keystr(p): leaf.shape for p, leaf in leaves_with_path}
    assert shapes[".weight"] == (4, 8)


def test_top_level_api_aliases():
    import xera.loom as L
    import xera.weave as W
    import xera.serialize as S

    assert xera.L is L
    assert xera.W is W
    assert xera.S is S
    assert xera.L is xera.loom
    assert xera.W is xera.weave
    assert xera.S is xera.serialize


def test_custom_module_with_loom():
    class MLP(loom.Module):
        in_features: int
        hidden: int
        out_features: int

        def setup(self):
            self.fc1 = loom.Dense(self.in_features, self.hidden, key=self.rng())
            self.fc2 = loom.Dense(self.hidden, self.out_features, key=self.rng())

        def __call__(self, x):
            return self.fc2(jax.nn.relu(self.fc1(x)))

    model = MLP(in_features=32, hidden=64, out_features=10, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 32))
    out = model(x)
    assert out.shape == (4, 10)


def test_module_without_key_cannot_call_rng():
    class NeedsRng(loom.Module):
        def setup(self):
            self.rng()  # no key was passed -> should raise

    try:
        NeedsRng()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "self.rng()" in str(e)


def test_rng_pool_split_returns_independent_keys():
    from xera.core import RNGPool

    pool = RNGPool(jax.random.PRNGKey(42))
    k1 = pool.next()
    k2 = pool.next()
    assert not jnp.array_equal(k1, k2)

    pool2 = RNGPool(jax.random.PRNGKey(42))
    keys = pool2.split(3)
    assert len(keys) == 3
    assert not jnp.array_equal(keys[0], keys[1])


def test_buffer_repr_and_pytree_leaf():
    from xera.core import Buffer

    buf = Buffer(jnp.array([1.0, 2.0]))
    assert "Buffer" in repr(buf)
    leaves = jax.tree_util.tree_leaves(buf)
    assert len(leaves) == 1
    assert jnp.allclose(leaves[0], jnp.array([1.0, 2.0]))
