

"""
Attention mechanisms for transformer and sequence models.

This module provides various attention implementations including multi-head
attention, grouped-query attention, and self-attention with support for
rotary position embeddings.
"""

from __future__ import annotations
import math
import jax
import jax.numpy as jnp
from .module import Module
from .linear import Linear
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


def alibi_slopes(num_heads):
    """
    Compute the per-head ALiBi slopes m_h from "Train Short, Test Long:
    Attention with Linear Biases Enables Input Length Extrapolation"
    (Press et al., 2021, https://arxiv.org/abs/2108.12409).

    For a power-of-2 `num_heads`, slopes follow a geometric sequence
    starting at 2**(-8/num_heads) and ratio 2**(-8/num_heads) (e.g. for 8
    heads: 1/2, 1/4, ..., 1/256). For other head counts, the sequence is
    extended by interleaving slopes from the next power of 2, as
    described in the paper's appendix.

    Args:
        num_heads: Number of attention heads.

    Returns:
        A float32 array of shape (num_heads,), one slope per head, in the
        same head order as the rest of this module's (B, H, T, D) layout.
    """
    def _slopes_power_of_2(n):
        start = 2.0 ** (-8.0 / n)
        return [start * (start ** i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = _slopes_power_of_2(num_heads)
    else:
        closest_pow2 = 2 ** math.floor(math.log2(num_heads))
        slopes = _slopes_power_of_2(closest_pow2)
        extra = alibi_slopes(2 * closest_pow2)[0::2][: num_heads - closest_pow2]
        slopes = slopes + list(extra)
    return jnp.array(slopes, dtype=jnp.float32)


def alibi_bias(num_heads, seq_len_q, seq_len_k=None):
    """
    Build the ALiBi additive attention bias for all heads.

    ALiBi (Press et al., 2021) adds a static, non-learned penalty to the
    pre-softmax attention logits, proportional to the distance between
    query and key positions, scaled by a head-specific slope (see
    `alibi_slopes`). Unlike RoPE, this acts *after* the Q/K dot product --
    it never touches Q/K themselves -- so it composes cleanly with RoPE
    (or any other pre-dot-product positional scheme) applied on the same
    layer; the two are not mutually exclusive.

    Args:
        num_heads: Number of attention heads.
        seq_len_q: Query sequence length.
        seq_len_k: Key sequence length. Defaults to `seq_len_q` (the
            common self-attention case). Pass this explicitly for
            cross-attention, where query and key sequence lengths differ.

    Returns:
        A float32 array of shape (num_heads, seq_len_q, seq_len_k), ready
        to broadcast against (batch, num_heads, seq_len_q, seq_len_k)
        attention logits -- e.g. via `bias[None]` when passing it to
        `flash_sdpa`, or added directly to `scores` in this module's
        manual einsum-based layers.
    """
    if seq_len_k is None:
        seq_len_k = seq_len_q
    slopes = alibi_slopes(num_heads)                          # (H,)
    pos_q = jnp.arange(seq_len_q)[:, None]                    # (Tq, 1)
    pos_k = jnp.arange(seq_len_k)[None, :]                    # (1, Tk)
    rel_pos = jnp.abs(pos_k - pos_q).astype(jnp.float32)      # (Tq, Tk)
    return -slopes[:, None, None] * rel_pos[None]             # (H, Tq, Tk)


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
        use_alibi: Whether to add ALiBi (Press et al., 2021) attention
            bias (default: False). Unlike `use_rope`, this doesn't touch
            Q/K -- it adds a static distance penalty to the logits after
            the dot product -- so it composes with `use_rope`: both may
            be True at once. This is not a standard combination and
            hasn't been studied in the literature the way either
            technique alone has, but there is nothing that prevents it
            mechanically: RoPE and ALiBi act at different points in the
            computation (pre- vs post-dot-product) and don't interfere
            structurally. Empirically, RoPE alone is typically more
            accurate within its training length but degrades sharply
            when extrapolated beyond it; ALiBi alone is typically less
            accurate at training length but extrapolates far more
            gracefully. Whether combining them keeps RoPE's in-length
            accuracy while inheriting ALiBi's extrapolation robustness is
            worth testing empirically for your own setup rather than
            assumed -- if you try it, treat this as a genuine experiment.
    
    Example:
        >>> import xera.loom as xl
        >>> attn = xl.MultiHeadAttention(dim=512, num_heads=8, dropout_rate=0.1)
        >>> output = attn(input_tensor, mask=causal_mask(seq_len))
        >>> # ALiBi instead of RoPE:
        >>> attn = xl.MultiHeadAttention(dim=512, num_heads=8, use_alibi=True)
        >>> # Both at once -- see the `use_alibi` note above:
        >>> attn = xl.MultiHeadAttention(dim=512, num_heads=8, use_rope=True, use_alibi=True)
    """
    
    dim: int
    num_heads: int
    dropout_rate: float = 0.0
    use_rope: bool = False
    rope_base: float = 10000.0
    use_alibi: bool = False

    def setup(self):
        """Initialize the attention projections and optional rotary embeddings."""
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = self.dim // self.num_heads
        
        self.q_proj = Linear(self.dim, self.dim, key=self.rng())
        self.k_proj = Linear(self.dim, self.dim, key=self.rng())
        self.v_proj = Linear(self.dim, self.dim, key=self.rng())
        self.out_proj = Linear(self.dim, self.dim, key=self.rng())
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
        if self.use_alibi:
            scores = scores + alibi_bias(H, T, T)[None]
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
        use_alibi: Whether to add ALiBi attention bias (default: False).
            See `MultiHeadAttention`'s `use_alibi` docstring for the
            RoPE+ALiBi combination notes -- they apply identically here.
    
    Example:
        >>> import xera.loom as xl
        >>> attn = xl.GroupedQueryAttention(dim=512, num_heads=8, num_kv_heads=2)
        >>> output = attn(input_tensor)
    """

    dim: int
    num_heads: int
    num_kv_heads: int
    dropout_rate: float = 0.0
    use_rope: bool = False
    rope_base: float = 10000.0
    use_alibi: bool = False

    def setup(self):
        """Initialize the attention projections with grouped-query configuration."""
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads"
        )
        self.head_dim = self.dim // self.num_heads
        kv_dim = self.head_dim * self.num_kv_heads

        self.q_proj = Linear(self.dim, self.dim, key=self.rng())
        self.k_proj = Linear(self.dim, kv_dim, key=self.rng())
        self.v_proj = Linear(self.dim, kv_dim, key=self.rng())
        self.out_proj = Linear(self.dim, self.dim, key=self.rng())
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
        if self.use_alibi:
            scores = scores + alibi_bias(H, T, T)[None]
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
        >>> import xera.loom as xl
        >>> attn = xl.SelfAttention(dim=256, dropout_rate=0.1)
        >>> output = attn(input_tensor, context=encoder_output)
    """

    dim: int
    dropout_rate: float = 0.0

    def setup(self):
        """Initialize the attention projections."""
        self.q_proj = Linear(self.dim, self.dim, key=self.rng())
        self.k_proj = Linear(self.dim, self.dim, key=self.rng())
        self.v_proj = Linear(self.dim, self.dim, key=self.rng())
        self.out_proj = Linear(self.dim, self.dim, key=self.rng())
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


__all__ = [
    "MultiHeadAttention", "GroupedQueryAttention", "SelfAttention",
    "causal_mask", "alibi_bias", "alibi_slopes",
]
