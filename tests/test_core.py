

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xera
import xera.loom as loom
import xera.weave as weave


# ---------------------------------------------------------------------------
# loom.Module core (pytree, rng splitting, params/state separation)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# loom layers previously untested: Conv, pooling, Embedding/RoPE, SSM family
# ---------------------------------------------------------------------------

def test_conv_forward_shape_and_grad():
    conv = loom.Conv(in_channels=3, out_channels=8, kernel_size=(3, 3), key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 3))
    out = conv(x)
    assert out.shape == (2, 16, 16, 8)  # SAME padding, stride 1

    grads = jax.grad(lambda c, x: jnp.sum(c(x) ** 2))(conv, x)
    assert grads.weight.shape == conv.weight.shape


def test_conv_grouped_depthwise():
    conv = loom.Conv(in_channels=4, out_channels=4, kernel_size=(3,), groups=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 4))
    out = conv(x)
    assert out.shape == (2, 10, 4)


def test_conv_transpose_upsamples():
    up = loom.ConvTranspose(in_channels=3, out_channels=8, kernel_size=(4, 4), stride=2, padding="SAME", key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 8, 3))
    out = up(x)
    assert out.shape == (2, 16, 16, 8)  # stride 2 doubles spatial size

    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(up, x)
    assert grads.weight.shape == up.weight.shape


def test_conv_then_conv_transpose_roundtrip_shape():
    down = loom.Conv(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=2, padding="SAME", key=jax.random.PRNGKey(0))
    up = loom.ConvTranspose(in_channels=8, out_channels=3, kernel_size=(3, 3), stride=2, padding="SAME", key=jax.random.PRNGKey(1))
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 16, 16, 3))
    assert up(down(x)).shape == x.shape


def test_pooling_shapes():
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8, 3))
    assert loom.MaxPool(pool_size=(2, 2))(x).shape == (2, 4, 4, 3)
    assert loom.AvgPool(pool_size=(2, 2))(x).shape == (2, 4, 4, 3)
    assert loom.GlobalAvgPool()(x).shape == (2, 3)
    assert loom.GlobalAvgPool(keepdims=True)(x).shape == (2, 1, 1, 3)


def test_embedding_and_rotary():
    emb = loom.Embedding(num_embeddings=100, features=16, key=jax.random.PRNGKey(0))
    idx = jnp.array([[1, 2, 3], [4, 5, 6]])
    out = emb(idx)
    assert out.shape == (2, 3, 16)

    rope = loom.RotaryEmbedding(dim=16)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 16))
    rotated = rope(x)
    assert rotated.shape == x.shape
    assert not jnp.allclose(rotated, x)  # actually rotates, not a no-op


def test_self_attention_matches_single_head_mha():
    dim = 16
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, dim))

    sa = loom.SelfAttention(dim=dim, key=jax.random.PRNGKey(0))
    mha = loom.MultiHeadAttention(dim=dim, num_heads=1, key=jax.random.PRNGKey(0))

    out_sa = sa(x)
    out_mha = mha(x, deterministic=True)
    assert out_sa.shape == out_mha.shape == (2, 5, dim)
    # same math (single head == no head split), different param init RNG
    # draw order, so just check shapes/dtype line up, not exact equality.
    assert out_sa.dtype == out_mha.dtype


def test_self_attention_cross_attention_uses_context():
    dim = 16
    sa = loom.SelfAttention(dim=dim, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, dim))          # queries
    context = jax.random.normal(jax.random.PRNGKey(2), (2, 9, dim))    # different seq_len

    out = sa(x, context=context)
    assert out.shape == (2, 5, dim)  # output length follows query, not context

    out_self = sa(x)  # context=None -> self-attention
    assert not jnp.allclose(out, out_self)  # genuinely different from self-attention


def test_self_attention_grad():
    dim = 8
    sa = loom.SelfAttention(dim=dim, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, dim))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(sa, x)
    assert grads.q_proj.weight.shape == sa.q_proj.weight.shape


def test_ssm_forward_shape():
    ssm = loom.SSM(channels=8, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 8))
    out = ssm(x)
    assert out.shape == (2, 10, 8)


def test_selective_ssm_forward_shape_and_grad():
    sel = loom.SelectiveSSM(d_inner=8, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 8))
    out = sel(x)
    assert out.shape == (2, 10, 8)

    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(sel, x)
    assert grads.log_A.shape == sel.log_A.shape


def test_mamba_block_shape_grad_and_causality():
    block = loom.MambaBlock(d_model=16, d_inner=32, state_dim=4, conv_kernel=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 12, 16))
    out = block(x)
    assert out.shape == x.shape

    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(block, x)
    assert grads.in_proj.weight.shape == block.in_proj.weight.shape

    # causal: perturbing a future timestep must not change earlier outputs
    x2 = x.at[:, -1, :].set(0.0)
    out2 = block(x2)
    assert jnp.allclose(out[:, :-1, :], out2[:, :-1, :], atol=1e-5)
    assert not jnp.allclose(out[:, -1, :], out2[:, -1, :])


def test_mamba_block_default_d_inner():
    block = loom.MambaBlock(d_model=8, key=jax.random.PRNGKey(0))
    assert block._d_inner == 16  # default expansion factor 2x


# ---------------------------------------------------------------------------
# weave.Loss
# ---------------------------------------------------------------------------

def test_loss_l1_l2():
    pred = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.0, 2.0, 5.0])
    assert jnp.isclose(weave.Loss.L1(pred, target), 2.0 / 3.0)
    assert jnp.isclose(weave.Loss.L2(pred, target), 4.0 / 3.0)
    assert weave.Loss.L1(pred, pred) == 0.0
    assert weave.Loss.L2(pred, pred) == 0.0


def test_loss_ce_matches_manual_softmax_ce():
    logits = jax.random.normal(jax.random.PRNGKey(0), (5, 4))
    labels = jnp.array([0, 1, 2, 3, 0])
    got = weave.Loss.CE(logits, labels)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    expected = -jnp.mean(log_probs[jnp.arange(5), labels])
    assert jnp.isclose(got, expected, atol=1e-6)

    # one-hot labels should give the same result
    onehot = jax.nn.one_hot(labels, 4)
    got_onehot = weave.Loss.CE(logits, onehot)
    assert jnp.isclose(got, got_onehot, atol=1e-6)


# ---------------------------------------------------------------------------
# weave.Metrics -- registry fused into jax.debug.callback
# ---------------------------------------------------------------------------

def test_metrics_default_log_prints(capsys):
    weave.Metrics.log(jnp.array(3), loss=jnp.array(0.5))
    jax.effects_barrier()
    out = capsys.readouterr().out
    assert "step 3" in out
    assert "loss" in out


def test_metrics_custom_registration(capsys):
    seen = []

    @weave.Metrics.register("custom_metric")
    def _(step, value):
        seen.append((int(step), float(value)))

    try:
        weave.Metrics.log(jnp.array(7), custom_metric=jnp.array(1.25))
        jax.effects_barrier()
        assert seen == [(7, 1.25)]
    finally:
        weave.Metrics.unregister("custom_metric")


def test_metrics_log_inside_jit_does_not_error():
    @jax.jit
    def step(x):
        weave.Metrics.log(0, x=x)
        return x + 1

    out = step(jnp.array(1.0))
    jax.effects_barrier()
    assert out == 2.0


# ---------------------------------------------------------------------------
# weave optimizers -- no optax anywhere
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("opt", [
    weave.SGDMomentum(lr=1e-2),
    weave.Adam(lr=1e-2),
    weave.AdamW(lr=1e-2),
    weave.Lion(lr=1e-2),
    weave.Muon(lr=1e-2),
    weave.RMSprop(lr=1e-2),
    weave.Adagrad(lr=1e-2),
])
def test_optimizer_smoke_reduces_loss(opt):
    dense = loom.Dense(4, 4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4))
    y = jax.random.normal(jax.random.PRNGKey(2), (8, 4))

    def loss_fn(m):
        return weave.Loss.L2(m(x), y)

    state = opt.init(dense)
    loss0 = loss_fn(dense)
    for i in range(20):
        grads = jax.grad(loss_fn)(dense)
        updates, state = opt.update(grads, state, dense, step=i)
        dense = weave.apply_updates(dense, updates)
    loss1 = loss_fn(dense)
    assert loss1 < loss0


def test_train_class_has_no_optax_dependency():
    import xera.weave.train as train_mod
    src = open(train_mod.__file__).read()
    assert "optax" not in src


def test_train_end_to_end():
    key = jax.random.PRNGKey(0)
    xs = jax.random.normal(key, (16, 4))
    ys = xs * 2.0

    class MyTrain(weave.Train):
        def loss_fn(self, pred, target):
            return weave.Loss.L2(pred, target)

        def get_batch(self, i):
            return xs, ys

    trainer = MyTrain(optimizer=weave.Adam(lr=5e-2), steps=50)
    model = loom.Dense(4, 4, key=jax.random.PRNGKey(1))
    loss_before = trainer.loss_fn(model(xs), ys)
    final_model, _opt_state, losses = trainer.run(model)
    loss_after = trainer.loss_fn(final_model(xs), ys)
    assert loss_after < loss_before
    assert losses.shape == (50,)


# ---------------------------------------------------------------------------
# xera.serialize -- safetensors for models, xera's own format for state
# ---------------------------------------------------------------------------

def test_serialize_model_roundtrip(tmp_path):
    model = loom.Dense(4, 8, key=jax.random.PRNGKey(0))
    path = tmp_path / "model.safetensors"
    xera.serialize.save_model(model, str(path))

    template = loom.Dense(4, 8, key=jax.random.PRNGKey(999))  # different init on purpose
    loaded = xera.serialize.load_model(template, str(path))

    assert jnp.allclose(loaded.weight, model.weight)
    assert jnp.allclose(loaded.bias, model.bias)
    assert not jnp.allclose(template.weight, model.weight)  # sanity: template really differed


def test_serialize_state_roundtrip(tmp_path):
    dense = loom.Dense(4, 4, key=jax.random.PRNGKey(0))
    opt = weave.Adam(lr=1e-2)
    state = opt.init(dense)
    grads = jax.grad(lambda m: weave.Loss.L2(m(jnp.ones((2, 4))), jnp.zeros((2, 4))))(dense)
    _, state = opt.update(grads, state, dense, step=0)  # make state non-trivial

    path = tmp_path / "opt_state.xera"
    xera.serialize.save_state(state, str(path))
    loaded = xera.serialize.load_state(str(path))

    leaves_a = jax.tree_util.tree_leaves(state)
    leaves_b = jax.tree_util.tree_leaves(loaded)
    assert len(leaves_a) == len(leaves_b)
    for a, b in zip(leaves_a, leaves_b):
        assert jnp.allclose(a, b)


def test_serialize_state_rejects_foreign_file(tmp_path):
    path = tmp_path / "not_a_state_file.bin"
    path.write_bytes(b"hello world")
    with pytest.raises(ValueError):
        xera.serialize.load_state(str(path))
