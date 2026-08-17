

"""
Transformer building blocks for sequence modeling.

This module provides the fundamental components of transformer architectures:
feed-forward networks (MLP) and transformer blocks combining self-attention
with feed-forward layers.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from ..core import Module
from .linear import Dense
from .normalization import LayerNorm
from .stochastic import Dropout
from .attention import MultiHeadAttention


class MLP(Module):
    """
    Feed-forward network (MLP) used in transformer architectures.
    
    A two-layer feed-forward network with GELU activation and dropout,
    commonly used as the feed-forward component in transformer blocks.
    
    Attributes:
        dim: Input and output dimension.
        hidden_dim: Hidden layer dimension (typically larger than dim).
        dropout_rate: Dropout rate for the hidden layer (default: 0.0).
    
    Example:
        >>> mlp = MLP(dim=512, hidden_dim=2048, dropout_rate=0.1)
        >>> output = mlp(input_tensor, key=dropout_key, deterministic=False)
    """
    
    dim: int
    hidden_dim: int
    dropout_rate: float = 0.0

    def setup(self):
        """Initialize the feed-forward network layers."""
        self.fc1 = Dense(self.dim, self.hidden_dim, key=self.rng())
        self.fc2 = Dense(self.hidden_dim, self.dim, key=self.rng())
        self.dropout = Dropout(self.dropout_rate, key=self.rng())

    def __call__(self, x, *, key=None, deterministic=True):
        """
        Apply the feed-forward network to the input.
        
        Args:
            x: Input tensor of shape (..., dim).
            key: Random key for dropout (required if not deterministic).
            deterministic: If True, disables dropout.
        
        Returns:
            Output tensor of shape (..., dim).
        """
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, key=key, deterministic=deterministic)
        x = self.fc2(x)
        return x


class TransformerBlock(Module):
    """
    Transformer block combining self-attention and feed-forward layers.
    
    Implements the standard transformer block with pre-layer normalization,
    residual connections, and dropout. This is the building block used in
    models like GPT, BERT, and Vision Transformers.
    
    Attributes:
        dim: Model dimension.
        num_heads: Number of attention heads.
        mlp_hidden_dim: Hidden dimension for the feed-forward network.
        dropout_rate: Dropout rate for attention and MLP (default: 0.0).
    
    Example:
        >>> block = TransformerBlock(dim=512, num_heads=8, mlp_hidden_dim=2048)
        >>> output = block(input_tensor, mask=causal_mask(seq_len), key=dropout_key)
    """
    
    dim: int
    num_heads: int
    mlp_hidden_dim: int
    dropout_rate: float = 0.0

    def setup(self):
        """Initialize the transformer block components."""
        self.attn = MultiHeadAttention(self.dim, self.num_heads, self.dropout_rate, key=self.rng())
        self.mlp = MLP(self.dim, self.mlp_hidden_dim, self.dropout_rate, key=self.rng())
        self.ln1 = LayerNorm(self.dim, key=self.rng())
        self.ln2 = LayerNorm(self.dim, key=self.rng())

    def __call__(self, x, *, mask=None, key=None, deterministic=True):
        """
        Apply the transformer block to the input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            mask: Optional attention mask for causal or padding masking.
            key: Random key for dropout (required if not deterministic).
            deterministic: If True, disables dropout.
        
        Returns:
            Output tensor of shape (batch, seq_len, dim).
        """
        if key is not None:
            k_attn, k_mlp = jax.random.split(key)
        else:
            k_attn = k_mlp = None

        x = x + self.attn(self.ln1(x), mask=mask, key=k_attn, deterministic=deterministic)
        x = x + self.mlp(self.ln2(x), key=k_mlp, deterministic=deterministic)
        return x


__all__ = ["MLP", "TransformerBlock"]