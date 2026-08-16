

from __future__ import annotations
import jax.numpy as jnp
from ..core import Module, param
from .. import initializers


class Embedding(Module):
    """Lookup-table embedding: integer indices -> learned vectors."""
    num_embeddings: int
    features: int

    def setup(self):
        self.weight = param(
            self.rng(), initializers.normal(stddev=0.02),
            (self.num_embeddings, self.features),
        )

    def __call__(self, idx):
        return self.weight[idx]


class RotaryEmbedding(Module):
    """Rotary Position Embedding (RoPE), Su et al. 2021:
    https://arxiv.org/abs/2104.09864

    Rotates pairs of elements along the last axis by an angle that grows
    with sequence position, so the dot product between a rotated query and
    key at positions `(i, j)` depends only on their relative offset `i-j`
    (not their absolute positions) -- the standard positional scheme in
    current transformer stacks (LLaMA, Mistral, etc.), and why
    `MultiHeadAttention`/`GroupedQueryAttention` below wire this in as an
    optional `use_rope` flag rather than expecting a separate absolute
    positional `Embedding` added to the input.

    No learnable params -- the rotation frequencies are a fixed schedule
    from `dim` and `base`, not trained.
    """
    dim: int
    base: float = 10000.0

    def __call__(self, x, *, offset=0):
        # x: (..., seq_len, dim)
        seq_len = x.shape[-2]
        inv_freq = 1.0 / (
            self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
        )
        t = jnp.arange(seq_len, dtype=jnp.float32) + offset
        freqs = jnp.outer(t, inv_freq)                     # (seq_len, dim//2)
        emb = jnp.concatenate([freqs, freqs], axis=-1)       # (seq_len, dim)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        return self._rotate(x, cos, sin)

    @staticmethod
    def _rotate(x, cos, sin):
        x1, x2 = jnp.split(x, 2, axis=-1)
        rotated_half = jnp.concatenate([-x2, x1], axis=-1)
        return x * cos + rotated_half * sin


__all__ = ["Embedding", "RotaryEmbedding"]
