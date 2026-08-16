

from __future__ import annotations
import jax
import jax.numpy as jnp
from ..core import Module
from .linear import Dense
from .stochastic import Dropout
from .embedding import RotaryEmbedding


def causal_mask(seq_len):
    """Boolean causal mask of shape `(seq_len, seq_len)`: `True` where a
    query position may attend to a key position (`key_pos <= query_pos`).
    Broadcasts against `MultiHeadAttention`/`GroupedQueryAttention`'s
    `(batch, heads, seq_len, seq_len)` score tensor -- pass directly as
    `mask=causal_mask(T)`.
    """
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


class MultiHeadAttention(Module):
    
    dim: int
    num_heads: int
    dropout_rate: float = 0.0
    use_rope: bool = False
    rope_base: float = 10000.0

    def setup(self):
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = self.dim // self.num_heads
        
        self.q_proj = Dense(self.dim, self.dim, key=self.rng())
        self.k_proj = Dense(self.dim, self.dim, key=self.rng())
        self.v_proj = Dense(self.dim, self.dim, key=self.rng())
        self.out_proj = Dense(self.dim, self.dim, key=self.rng())
        self.dropout = Dropout(self.dropout_rate, key=self.rng())
        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, self.rope_base)

    def __call__(self, x, *, mask=None, key=None, deterministic=True):
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(D)
        if mask is not None:
            scores = jnp.where(mask, scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn, key=key, deterministic=deterministic)

        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.dim)
        return self.out_proj(out)


class GroupedQueryAttention(Module):
    """Grouped-query attention (Ainslie et al. 2023):
    https://arxiv.org/abs/2305.13245

    Like `MultiHeadAttention`, but K/V are projected to fewer heads than
    Q -- each KV head is shared across `num_heads // num_kv_heads` query
    heads, shrinking the KV-cache (the usual bottleneck for long-context
    autoregressive inference) without shrinking the number of query heads.
    `num_kv_heads=1` recovers multi-query attention (MQA, Shazeer 2019);
    `num_kv_heads=num_heads` recovers ordinary `MultiHeadAttention` (in
    fact, identically -- same math, just via the repeat-then-attend path
    with a group size of 1).
    """
    dim: int
    num_heads: int
    num_kv_heads: int
    dropout_rate: float = 0.0
    use_rope: bool = False
    rope_base: float = 10000.0

    def setup(self):
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads"
        )
        self.head_dim = self.dim // self.num_heads
        kv_dim = self.head_dim * self.num_kv_heads

        self.q_proj = Dense(self.dim, self.dim, key=self.rng())
        self.k_proj = Dense(self.dim, kv_dim, key=self.rng())
        self.v_proj = Dense(self.dim, kv_dim, key=self.rng())
        self.out_proj = Dense(self.dim, self.dim, key=self.rng())
        self.dropout = Dropout(self.dropout_rate, key=self.rng())
        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, self.rope_base)

    def __call__(self, x, *, mask=None, key=None, deterministic=True):
        B, T, _ = x.shape
        H, KVH, D = self.num_heads, self.num_kv_heads, self.head_dim
        group = H // KVH

        q = self.q_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)      # (B,H,T,D)
        k = self.k_proj(x).reshape(B, T, KVH, D).transpose(0, 2, 1, 3)    # (B,KVH,T,D)
        v = self.v_proj(x).reshape(B, T, KVH, D).transpose(0, 2, 1, 3)    # (B,KVH,T,D)

        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        k = jnp.repeat(k, group, axis=1)  # (B,H,T,D) -- each KV head shared by `group` Q heads
        v = jnp.repeat(v, group, axis=1)

        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(D)
        if mask is not None:
            scores = jnp.where(mask, scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn, key=key, deterministic=deterministic)

        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.dim)
        return self.out_proj(out)


__all__ = ["MultiHeadAttention", "GroupedQueryAttention", "causal_mask"]
