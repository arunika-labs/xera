

from __future__ import annotations
import jax
import jax.numpy as jnp
from ..core import Module
from .linear import Dense
from .stochastic import Dropout


class MultiHeadAttention(Module):
    
    dim: int
    num_heads: int
    dropout_rate: float = 0.0

    def setup(self):
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = self.dim // self.num_heads
        
        self.q_proj = Dense(self.dim, self.dim, key=self.rng())
        self.k_proj = Dense(self.dim, self.dim, key=self.rng())
        self.v_proj = Dense(self.dim, self.dim, key=self.rng())
        self.out_proj = Dense(self.dim, self.dim, key=self.rng())
        self.dropout = Dropout(self.dropout_rate, key=self.rng())

    def __call__(self, x, *, mask=None, key=None, deterministic=True):
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(D)
        if mask is not None:
            scores = jnp.where(mask, scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn, key=key, deterministic=deterministic)

        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.dim)
        return self.out_proj(out)


__all__ = ["MultiHeadAttention"]