

"""
Embedding layers for discrete inputs and positional encodings.

This module provides embedding layers for converting discrete indices to
continuous vectors, and rotary position embeddings for encoding
positional information in transformer models.
"""

from __future__ import annotations
import jax.numpy as jnp
from ..core import Module, param
from .. import initializers


class Embedding(Module):
    """
    Standard embedding layer for discrete inputs.
    
    Converts discrete indices (e.g., token IDs) to dense continuous vectors.
    Commonly used in NLP for word embeddings and in other domains for
    categorical feature embeddings.
    
    Attributes:
        num_embeddings: Size of the vocabulary (number of discrete indices).
        features: Dimension of the embedding vectors.
    
    Example:
        >>> embed = Embedding(num_embeddings=10000, features=256)
        >>> embeddings = embed(token_ids)  # shape: (batch, seq_len, 256)
    """

    num_embeddings: int
    features: int

    def setup(self):
        """Initialize the embedding weight matrix."""
        self.weight = param(
            self.rng(), initializers.normal(stddev=0.02),
            (self.num_embeddings, self.features),
        )

    def __call__(self, idx):
        """
        Look up embeddings for the given indices.
        
        Args:
            idx: Integer indices of shape (...) or (..., seq_len).
        
        Returns:
            Embedding vectors of shape (..., features) or (..., seq_len, features).
        """
        return self.weight[idx]


class RotaryEmbedding(Module):
    """
    Rotary Position Embedding (RoPE) for transformer models.
    
    RoPE encodes positional information by rotating query and key vectors
    in complex space. This is more effective than absolute position embeddings
    for extrapolating to longer sequences and maintaining relative position
    information.
    
    Attributes:
        dim: The dimension of the vectors to apply rotary embeddings to.
        base: The base for the frequency computation (default: 10000.0).
    
    Example:
        >>> rope = RotaryEmbedding(dim=64, base=10000.0)
        >>> rotated_q = rope(q)  # Apply to query vectors
        >>> rotated_k = rope(k)  # Apply to key vectors
    """

    dim: int
    base: float = 10000.0

    def __call__(self, x, *, offset=0):
        """
        Apply rotary position embeddings to the input.
        
        Args:
            x: Input tensor of shape (..., seq_len, dim).
            offset: Position offset for the embeddings (default: 0).
        
        Returns:
            Rotated tensor of the same shape as input.
        """
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
        """
        Apply rotation to the input vectors using cosine and sine components.
        
        Args:
            x: Input tensor to rotate.
            cos: Cosine components of the rotation.
            sin: Sine components of the rotation.
        
        Returns:
            Rotated tensor.
        """
        x1, x2 = jnp.split(x, 2, axis=-1)
        rotated_half = jnp.concatenate([-x2, x1], axis=-1)
        return x * cos + rotated_half * sin


__all__ = ["Embedding", "RotaryEmbedding"]
