

"""
Attention mechanisms for transformer and sequence models.

This module provides various attention implementations including multi-head
attention, grouped-query attention, and self-attention with support for
rotary position embeddings.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from .module import Module
from .linear import Dense
from .stochastic import Dropout
from .embedding import RotaryEmbedding


def causal_mask(seq_len):
    """
    Create a causal (lower triangular) attention mask.
    
    This mask prevents positions from attending to future positions,
    which is essential for autoregressive generation.
    
    Args:
        seq_len: The length of the sequence.
    
    Returns:
        A boolean mask of shape (seq_len, seq_len) where True indicates
        allowed attention positions (lower triangular).
    """
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


class MultiHeadAttention(Module):
    """
    Multi-head attention mechanism.
    
    This layer implements the standard multi-head attention from "Attention
    is All You Need". It splits the attention mechanism into multiple heads
    to allow the model to attend to different representation subspaces.
    
    Attributes:
        dim: The dimension of the input and output.
        num_heads: Number of attention heads. Must divide dim evenly.
        dropout_rate: Dropout rate for attention weights (default: 0.0).
        use_rope: Whether to use rotary position embeddings (default: False).
        rope_base: Base for rotary position embedding frequencies (default: 10000.0).
    
    Example:
        >>> attn = MultiHeadAttention(dim=512, num_heads=8, dropout_rate=0.1)
        >>> output = attn(input_tensor, mask=causal_mask(seq_len))
    """
    
    dim: int
    num_heads: int
    dropout_rate: float = 0.0
    use_rope: bool = False
    rope_base: float = 10000.0

    def setup(self):
        """Initialize the attention projections and optional rotary embeddings."""
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
        """
        Apply multi-head attention to the input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            mask: Optional attention mask. True for allowed positions.
            key: Optional random key for dropout.
            deterministic: If True, disables dropout.
        
        Returns:
            Output tensor of shape (batch, seq_len, dim).
        """
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
    """
    Grouped-query attention (GQA) mechanism.
    
    GQA is a memory-efficient variant of multi-head attention where multiple
    query heads share the same key and value heads. This reduces the memory
    and computation cost while maintaining most of the performance benefits.
    
    Attributes:
        dim: The dimension of the input and output.
        num_heads: Number of query heads. Must divide dim evenly.
        num_kv_heads: Number of key/value heads. Must divide num_heads.
        dropout_rate: Dropout rate for attention weights (default: 0.0).
        use_rope: Whether to use rotary position embeddings (default: False).
        rope_base: Base for rotary position embedding frequencies (default: 10000.0).
    
    Example:
        >>> attn = GroupedQueryAttention(dim=512, num_heads=8, num_kv_heads=2)
        >>> output = attn(input_tensor)
    """

    dim: int
    num_heads: int
    num_kv_heads: int
    dropout_rate: float = 0.0
    use_rope: bool = False
    rope_base: float = 10000.0

    def setup(self):
        """Initialize the attention projections with grouped-query configuration."""
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
        """
        Apply grouped-query attention to the input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            mask: Optional attention mask. True for allowed positions.
            key: Optional random key for dropout.
            deterministic: If True, disables dropout.
        
        Returns:
            Output tensor of shape (batch, seq_len, dim).
        """
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


class SelfAttention(Module):
    """
    Single-head self-attention mechanism.
    
    This is a simpler attention mechanism without head splitting, useful for
    smaller models or specific architectures. It can also serve as cross-attention
    when a context is provided.
    
    Attributes:
        dim: The dimension of the input and output.
        dropout_rate: Dropout rate for attention weights (default: 0.0).
    
    Example:
        >>> attn = SelfAttention(dim=256, dropout_rate=0.1)
        >>> output = attn(input_tensor, context=encoder_output)
    """

    dim: int
    dropout_rate: float = 0.0

    def setup(self):
        """Initialize the attention projections."""
        self.q_proj = Dense(self.dim, self.dim, key=self.rng())
        self.k_proj = Dense(self.dim, self.dim, key=self.rng())
        self.v_proj = Dense(self.dim, self.dim, key=self.rng())
        self.out_proj = Dense(self.dim, self.dim, key=self.rng())
        self.dropout = Dropout(self.dropout_rate, key=self.rng())

    def __call__(self, x, *, context=None, mask=None, key=None, deterministic=True):
        """
        Apply self-attention (or cross-attention if context is provided).
        
        Args:
            x: Query input tensor of shape (batch, query_len, dim).
            context: Optional key/value input for cross-attention.
                If None, uses x as both query and key/value.
            mask: Optional attention mask. True for allowed positions.
            key: Optional random key for dropout.
            deterministic: If True, disables dropout.
        
        Returns:
            Output tensor of shape (batch, query_len, dim).
        """
        kv_source = context if context is not None else x

        q = self.q_proj(x)                # (B, Tq, dim)
        k = self.k_proj(kv_source)         # (B, Tk, dim)
        v = self.v_proj(kv_source)         # (B, Tk, dim)

        scores = jnp.einsum("btd,bsd->bts", q, k) / jnp.sqrt(self.dim)
        if mask is not None:
            scores = jnp.where(mask, scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn, key=key, deterministic=deterministic)

        out = jnp.einsum("bts,bsd->btd", attn, v)
        return self.out_proj(out)


__all__ = ["MultiHeadAttention", "GroupedQueryAttention", "SelfAttention", "causal_mask"]
