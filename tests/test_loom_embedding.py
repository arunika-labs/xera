"""Tests for xera.loom.embedding: Embedding, RotaryEmbedding."""

import jax
import jax.numpy as jnp
import xera.loom as loom


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


def test_embedding_lookup_matches_table_row():
    emb = loom.Embedding(num_embeddings=10, features=4, key=jax.random.PRNGKey(0))
    idx = jnp.array([[3]])
    out = emb(idx)
    assert jnp.allclose(out[0, 0], emb.weight[3])


def test_embedding_grad_shape():
    emb = loom.Embedding(num_embeddings=10, features=4, key=jax.random.PRNGKey(0))
    idx = jnp.array([0, 1, 2])
    grads = jax.grad(lambda m, i: jnp.sum(m(i) ** 2))(emb, idx)
    assert grads.weight.shape == emb.weight.shape


def test_rotary_embedding_preserves_norm():
    # RoPE is a rotation, so it should preserve the vector norm per position.
    rope = loom.RotaryEmbedding(dim=8)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, 8))
    rotated = rope(x)
    norm_before = jnp.linalg.norm(x, axis=-1)
    norm_after = jnp.linalg.norm(rotated, axis=-1)
    assert jnp.allclose(norm_before, norm_after, atol=1e-4)
