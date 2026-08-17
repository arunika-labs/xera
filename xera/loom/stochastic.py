

"""
Stochastic layers for regularization during neural network training.

This module provides dropout layers that help prevent overfitting by
randomly dropping units during training.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from ..core import Module


class Dropout(Module):
    """
    Dropout layer for regularization.
    
    During training, randomly sets a fraction of input units to zero with
    probability `rate`. This helps prevent overfitting by reducing complex
    co-adaptations between neurons. During inference, the layer is bypassed
    but the output is scaled to maintain the expected value.
    
    Attributes:
        rate: Dropout probability (fraction of units to drop).
    
    Example:
        >>> dropout = Dropout(rate=0.1)
        >>> output = dropout(input_tensor, key=dropout_key, deterministic=False)
    """
    
    rate: float

    def setup(self):
        """No setup needed for Dropout."""
        pass  

    def __call__(self, x, *, key=None, deterministic=True):
        """
        Apply dropout to the input tensor.
        
        Args:
            x: Input tensor.
            key: Random key for generating dropout mask (required if not deterministic).
            deterministic: If True, bypass dropout (inference mode). If False,
                apply dropout (training mode).
        
        Returns:
            Output tensor with dropout applied during training, or scaled
            input during inference.
        """
        if deterministic or self.rate == 0.0:
            return x
        keep_prob = 1.0 - self.rate
        mask = jax.random.bernoulli(key, keep_prob, x.shape)
        return jnp.where(mask, x / keep_prob, 0.0)


__all__ = ["Dropout"]