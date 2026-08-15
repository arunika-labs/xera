

from __future__ import annotations
import jax
import jax.numpy as jnp
from ..core import Module
from .linear import Dense
from .normalization import LayerNorm
from .stochastic import Dropout
from .attention import MultiHeadAttention


class MLP(Module):
    
    dim: int
    hidden_dim: int
    dropout_rate: float = 0.0

    def setup(self):
        self.fc1 = Dense(self.dim, self.hidden_dim, key=self.rng())
        self.fc2 = Dense(self.hidden_dim, self.dim, key=self.rng())
        self.dropout = Dropout(self.dropout_rate, key=self.rng())

    def __call__(self, x, *, key=None, deterministic=True):
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, key=key, deterministic=deterministic)
        x = self.fc2(x)
        return x


class TransformerBlock(Module):
    
    dim: int
    num_heads: int
    mlp_hidden_dim: int
    dropout_rate: float = 0.0

    def setup(self):
        self.attn = MultiHeadAttention(self.dim, self.num_heads, self.dropout_rate, key=self.rng())
        self.mlp = MLP(self.dim, self.mlp_hidden_dim, self.dropout_rate, key=self.rng())
        self.ln1 = LayerNorm(self.dim, key=self.rng())
        self.ln2 = LayerNorm(self.dim, key=self.rng())

    def __call__(self, x, *, mask=None, key=None, deterministic=True):
        if key is not None:
            k_attn, k_mlp = jax.random.split(key)
        else:
            k_attn = k_mlp = None

        x = x + self.attn(self.ln1(x), mask=mask, key=k_attn, deterministic=deterministic)
        x = x + self.mlp(self.ln2(x), key=k_mlp, deterministic=deterministic)
        return x


__all__ = ["MLP", "TransformerBlock"]