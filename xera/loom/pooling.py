

"""
Pooling layers for spatial downsampling in neural networks.

This module provides common pooling operations used in convolutional
networks for spatial dimensionality reduction and feature extraction.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from .module import Module


def _as_tuple(v, n):
    """
    Convert a value to a tuple of length n if it isn't already.
    
    Args:
        v: Value to convert (single value or tuple).
        n: Length of the resulting tuple.
    
    Returns:
        A tuple of length n containing the value v repeated n times,
        or v if it's already a tuple.
    """
    return v if isinstance(v, tuple) else (v,) * n


class MaxPool(Module):
    """
    Max pooling layer for spatial downsampling.
    
    This layer applies max pooling over spatial dimensions, selecting the
    maximum value in each pooling window. This is commonly used in CNNs
    to reduce spatial dimensions while preserving the most salient features.
    
    Attributes:
        pool_size: Tuple of window sizes for each spatial dimension.
        stride: Tuple of strides for each spatial dimension. If None, uses pool_size.
        padding: Padding mode: "VALID" (no padding) or "SAME" (padding to maintain size).
    
    Example:
        >>> pool = MaxPool(pool_size=(2, 2), stride=(2, 2), padding="VALID")
        >>> output = pool(input_tensor)  # Halves spatial dimensions
    """

    pool_size: tuple
    stride: tuple = None
    padding: str = "VALID"

    def __call__(self, x):
        """
        Apply max pooling to the input tensor.
        
        Args:
            x: Input tensor of shape (batch, *spatial, channels).
        
        Returns:
            Pooled tensor with reduced spatial dimensions.
        """
        n_spatial = len(self.pool_size)
        stride = self.stride if self.stride is not None else self.pool_size
        window_dims = (1,) + tuple(self.pool_size) + (1,)
        window_strides = (1,) + _as_tuple(stride, n_spatial) + (1,)
        return jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, window_dims, window_strides, self.padding
        )


class AvgPool(Module):
    """
    Average pooling layer for spatial downsampling.
    
    This layer applies average pooling over spatial dimensions, computing
    the mean value in each pooling window. This is commonly used in CNNs
    to reduce spatial dimensions while providing a smoother downsampling
    than max pooling.
    
    Attributes:
        pool_size: Tuple of window sizes for each spatial dimension.
        stride: Tuple of strides for each spatial dimension. If None, uses pool_size.
        padding: Padding mode: "VALID" (no padding) or "SAME" (padding to maintain size).
    
    Example:
        >>> pool = AvgPool(pool_size=(2, 2), stride=(2, 2), padding="VALID")
        >>> output = pool(input_tensor)  # Halves spatial dimensions
    """

    pool_size: tuple
    stride: tuple = None
    padding: str = "VALID"

    def __call__(self, x):
        """
        Apply average pooling to the input tensor.
        
        Args:
            x: Input tensor of shape (batch, *spatial, channels).
        
        Returns:
            Pooled tensor with reduced spatial dimensions.
        """
        n_spatial = len(self.pool_size)
        stride = self.stride if self.stride is not None else self.pool_size
        window_dims = (1,) + tuple(self.pool_size) + (1,)
        window_strides = (1,) + _as_tuple(stride, n_spatial) + (1,)

        summed = jax.lax.reduce_window(
            x, 0.0, jax.lax.add, window_dims, window_strides, self.padding
        )
        counts = jax.lax.reduce_window(
            jnp.ones_like(x), 0.0, jax.lax.add, window_dims, window_strides, self.padding
        )
        return summed / counts


class GlobalAvgPool(Module):
    """
    Global average pooling layer.
    
    This layer computes the average over all spatial dimensions, reducing
    each spatial feature map to a single value. This is commonly used to
    convert convolutional feature maps to a global representation before
    fully connected layers.
    
    Attributes:
        keepdims: If True, keeps the reduced dimensions with size 1.
            If False, removes the reduced dimensions.
    
    Example:
        >>> pool = GlobalAvgPool(keepdims=False)
        >>> output = pool(input_tensor)  # Shape: (batch, channels)
    """

    keepdims: bool = False

    def __call__(self, x):
        """
        Apply global average pooling to the input tensor.
        
        Args:
            x: Input tensor of shape (batch, *spatial, channels).
        
        Returns:
            Tensor of shape (batch, channels) if keepdims=False,
            or (batch, 1, ..., 1, channels) if keepdims=True.
        """
        spatial_axes = tuple(range(1, x.ndim - 1))
        return jnp.mean(x, axis=spatial_axes, keepdims=self.keepdims)


__all__ = ["MaxPool", "AvgPool", "GlobalAvgPool"]
