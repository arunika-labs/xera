
import jax
import jax.numpy as jnp
import optax
import numpy as np
import xera
import xera.loom as loom


def test_forward_and_auto_rng_split():
    block = loom.TransformerBlock(dim=32, num_heads=4, mlp_hidden_dim=64, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 32))
    out = block(x, deterministic=True)
    assert out.shape == (2, 8, 32)
    assert not np.allclose(block.attn.q_proj.weight, block.attn.k_proj.weight)


def test_jit_and_grad():
    block = loom.TransformerBlock(dim=32, num_heads=4, mlp_hidden_dim=64, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 32))

    fwd = jax.jit(lambda b, x: b(x, deterministic=True))
    out = fwd(block, x)
    assert out.shape == (2, 8, 32)

    grads = jax.grad(lambda b, x: jnp.sum(fwd(b, x) ** 2))(block, x)
    assert grads.attn.q_proj.weight.shape == (32, 32)


def test_optax_training_step():
    block = loom.TransformerBlock(dim=16, num_heads=2, mlp_hidden_dim=32, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 16))
    y = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 16))

    opt = optax.adam(1e-3)
    opt_state = opt.init(block)

    def loss_fn(b, x, y):
        return jnp.mean((b(x, deterministic=True) - y) ** 2)

    loss0 = loss_fn(block, x, y)
    for _ in range(10):
        grads = jax.grad(loss_fn)(block, x, y)
        updates, opt_state = opt.update(grads, opt_state, block)
        block = optax.apply_updates(block, updates)
    loss1 = loss_fn(block, x, y)
    assert loss1 < loss0


def test_params_dict():
    dense = loom.Dense(4, 8, key=jax.random.PRNGKey(0))
    pd = dense.params_dict()
    assert set(pd.keys()) == {"weight", "bias"}
    assert pd["weight"].shape == (4, 8)


def test_batchnorm_state_separate_from_params():
    bn = loom.BatchNorm(dim=16, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 16))
    _, bn2 = bn(x, training=True)
    assert not jnp.allclose(bn.running_mean.value, bn2.running_mean.value)
    assert jnp.allclose(bn.gamma, bn2.gamma)
    assert jnp.allclose(bn.beta, bn2.beta)


def test_top_level_api_aliases():
    import xera.loom as L
    import xera.weave as W

    assert xera.L is L
    assert xera.W is W
    assert xera.L is xera.loom
    assert xera.W is xera.weave


def test_custom_module_with_loom():
    class MLP(xera.Module):
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
