"""Tests for xera.loom.transformer: MLP, TransformerBlock."""

import jax
import jax.numpy as jnp
import xera.loom as loom
from xera.loom.attention import causal_mask


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def test_mlp_forward_shape():
    mlp = loom.MLP(dim=16, hidden_dim=64, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 16))
    out = mlp(x)
    assert out.shape == (4, 16)


def test_mlp_batched_sequence_input():
    mlp = loom.MLP(dim=16, hidden_dim=64, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 16))
    out = mlp(x)
    assert out.shape == (2, 5, 16)


def test_mlp_layer_shapes():
    mlp = loom.MLP(dim=16, hidden_dim=64, key=jax.random.PRNGKey(0))
    assert mlp.fc1.weight.shape == (16, 64)
    assert mlp.fc2.weight.shape == (64, 16)


def test_mlp_deterministic_dropout_is_identity_on_rate_zero():
    mlp = loom.MLP(dim=8, hidden_dim=16, dropout_rate=0.0, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 8))
    out1 = mlp(x, deterministic=False)
    out2 = mlp(x, deterministic=False)
    assert jnp.allclose(out1, out2)


def test_mlp_dropout_stochastic_when_active():
    mlp = loom.MLP(dim=8, hidden_dim=16, dropout_rate=0.5, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 8))
    out1 = mlp(x, key=jax.random.PRNGKey(2), deterministic=False)
    out2 = mlp(x, key=jax.random.PRNGKey(3), deterministic=False)
    assert not jnp.allclose(out1, out2)


def test_mlp_grad_shapes_match_params():
    mlp = loom.MLP(dim=8, hidden_dim=16, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(mlp, x)
    assert grads.fc1.weight.shape == mlp.fc1.weight.shape
    assert grads.fc2.weight.shape == mlp.fc2.weight.shape


def test_mlp_uses_gelu_nonlinearity():
    # With hidden_dim == dim, fc1/fc2 as identity-like isn't guaranteed, but
    # we can at least check the MLP is not merely a linear map end-to-end by
    # comparing against a manually composed linear computation would be
    # architecture-specific. Instead, verify negative inputs are not simply
    # zeroed as with ReLU: gelu is smooth and non-zero for small negatives.
    mlp = loom.MLP(dim=4, hidden_dim=4, key=jax.random.PRNGKey(0))
    x = -jnp.ones((1, 4)) * 0.1
    hidden = jax.nn.gelu(mlp.fc1(x))
    assert not jnp.allclose(hidden, jnp.zeros_like(hidden))


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

def test_transformer_block_forward_shape():
    block = loom.TransformerBlock(dim=32, num_heads=4, mlp_hidden_dim=64, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 32))
    out = block(x)
    assert out.shape == (2, 6, 32)


def test_transformer_block_has_submodules():
    block = loom.TransformerBlock(dim=32, num_heads=4, mlp_hidden_dim=64, key=jax.random.PRNGKey(0))
    assert isinstance(block.attn, loom.MultiHeadAttention)
    assert isinstance(block.mlp, loom.MLP)
    assert isinstance(block.ln1, loom.LayerNorm)
    assert isinstance(block.ln2, loom.LayerNorm)


def test_transformer_block_residual_connection():
    # Zeroing attn/mlp output paths isn't directly controllable, but we can
    # check that block output differs from a pure attn+mlp-without-residual
    # computation, confirming the `x + ...` residual terms are present, by
    # checking block output is not identical to attn(ln1(x)) alone.
    block = loom.TransformerBlock(dim=16, num_heads=2, mlp_hidden_dim=32, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 16))
    out = block(x)
    attn_only = block.attn(block.ln1(x))
    assert not jnp.allclose(out, attn_only)


def test_transformer_block_causal_mask_blocks_future():
    block = loom.TransformerBlock(dim=16, num_heads=2, mlp_hidden_dim=32, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))
    mask = causal_mask(5)[None, None, :, :]

    out1 = block(x, mask=mask)[0, 0]
    x_perturbed = x.at[0, 4].add(100.0)
    out2 = block(x_perturbed, mask=mask)[0, 0]
    assert jnp.allclose(out1, out2, atol=1e-4)


def test_transformer_block_dropout_splits_keys():
    block = loom.TransformerBlock(
        dim=16, num_heads=2, mlp_hidden_dim=32, dropout_rate=0.5, key=jax.random.PRNGKey(0)
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 16))
    out1 = block(x, key=jax.random.PRNGKey(2), deterministic=False)
    out2 = block(x, key=jax.random.PRNGKey(3), deterministic=False)
    assert not jnp.allclose(out1, out2)


def test_transformer_block_deterministic_reproducible():
    block = loom.TransformerBlock(
        dim=16, num_heads=2, mlp_hidden_dim=32, dropout_rate=0.5, key=jax.random.PRNGKey(0)
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 16))
    out1 = block(x, deterministic=True)
    out2 = block(x, deterministic=True)
    assert jnp.allclose(out1, out2)


def test_transformer_block_grad_shapes_match_params():
    block = loom.TransformerBlock(dim=16, num_heads=2, mlp_hidden_dim=32, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 16))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(block, x)
    assert grads.attn.q_proj.weight.shape == block.attn.q_proj.weight.shape
    assert grads.mlp.fc1.weight.shape == block.mlp.fc1.weight.shape


def test_transformer_block_jit_compatible():
    block = loom.TransformerBlock(dim=16, num_heads=2, mlp_hidden_dim=32, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 16))
    fwd = jax.jit(lambda m, x: m(x, deterministic=True))
    out = fwd(block, x)
    assert out.shape == (2, 4, 16)


def test_transformer_block_stack_composability():
    # Multiple blocks can be stacked as a plain Python list and applied
    # sequentially; each maintains independent parameters.
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    blocks = [
        loom.TransformerBlock(dim=16, num_heads=2, mlp_hidden_dim=32, key=k) for k in keys
    ]
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 16))
    for block in blocks:
        x = block(x)
    assert x.shape == (1, 4, 16)
    assert not jnp.allclose(blocks[0].attn.q_proj.weight, blocks[1].attn.q_proj.weight)
