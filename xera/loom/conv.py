

"""
Convolutional layers for spatial feature extraction in neural networks.

This module provides standard convolution and transposed convolution layers
with support for various configurations including strided convolutions,
dilated convolutions, and grouped convolutions.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from .module import Module, param
from . import initializers


def _dimension_numbers(ndim: int) -> jax.lax.ConvDimensionNumbers:
    """
    Compute dimension numbers for JAX convolution operations.
    
    This sets up the dimension ordering for convolution operations to match
    the framework's convention: (batch, *spatial, channels).
    
    Args:
        ndim: The total number of dimensions in the input tensor.
    
    Returns:
        A ConvDimensionNumbers object specifying the dimension layout.
    """
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


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


class Conv(Module):
    """
    Standard convolution layer.
    
    Applies a convolution operation over spatial dimensions of the input.
    Supports strided convolutions, dilated convolutions, and grouped convolutions.
    
    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Tuple of kernel sizes for each spatial dimension.
        stride: Stride for each spatial dimension (default: 1).
        padding: Padding mode: "SAME", "VALID", or a tuple of padding values.
        dilation: Dilation factor for each spatial dimension (default: 1).
        groups: Number of groups for grouped convolution (default: 1).
        use_bias: Whether to add a bias term (default: True).
    
    Example:
        >>> conv = Conv(in_channels=3, out_channels=64, kernel_size=(3, 3))
        >>> output = conv(input_tensor)
    """

    in_channels: int
    out_channels: int
    kernel_size: tuple
    stride: int | tuple = 1
    padding: str | tuple = "SAME"
    dilation: int | tuple = 1
    groups: int = 1
    use_bias: bool = True

    def setup(self):
        """Initialize convolution weights and optional bias."""
        assert self.in_channels % self.groups == 0, (
            f"in_channels ({self.in_channels}) must be divisible by groups ({self.groups})"
        )
        assert self.out_channels % self.groups == 0, (
            f"out_channels ({self.out_channels}) must be divisible by groups ({self.groups})"
        )

        weight_shape = tuple(self.kernel_size) + (self.in_channels // self.groups, self.out_channels)
        self.weight = param(self.rng(), initializers.kaiming_normal(), weight_shape)
        self.bias = (
            param(self.rng(), initializers.zeros(), (self.out_channels,))
            if self.use_bias else None
        )

    def __call__(self, x):
        """
        Apply convolution to the input tensor.
        
        Args:
            x: Input tensor of shape (batch, *spatial, in_channels).
        
        Returns:
            Output tensor of shape (batch, *spatial', out_channels).
        """
        n_spatial = len(self.kernel_size)
        stride = _as_tuple(self.stride, n_spatial)
        dilation = _as_tuple(self.dilation, n_spatial)
        padding = self.padding if isinstance(self.padding, str) else tuple(self.padding)

        y = jax.lax.conv_general_dilated(
            x, self.weight,
            window_strides=stride,
            padding=padding,
            rhs_dilation=dilation,
            dimension_numbers=_dimension_numbers(x.ndim),
            feature_group_count=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y


class ConvTranspose(Module):
    """
    Transposed convolution (fractionally-strided convolution) layer.
    
    Also known as deconvolution, this layer performs the inverse operation
    of a standard convolution, commonly used for upsampling in generative
    models and segmentation networks.
    
    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Tuple of kernel sizes for each spatial dimension.
        stride: Stride for each spatial dimension (default: 1).
        padding: Padding mode: "SAME", "VALID", or a tuple of padding values.
        dilation: Dilation factor for each spatial dimension (default: 1).
        use_bias: Whether to add a bias term (default: True).
    
    Example:
        >>> deconv = ConvTranspose(in_channels=64, out_channels=3, kernel_size=(3, 3))
        >>> output = deconv(input_tensor)
    """

    in_channels: int
    out_channels: int
    kernel_size: tuple
    stride: int | tuple = 1
    padding: str | tuple = "SAME"
    dilation: int | tuple = 1
    use_bias: bool = True

    def setup(self):
        """Initialize transposed convolution weights and optional bias."""
        weight_shape = tuple(self.kernel_size) + (self.in_channels, self.out_channels)
        self.weight = param(self.rng(), initializers.kaiming_normal(), weight_shape)
        self.bias = (
            param(self.rng(), initializers.zeros(), (self.out_channels,))
            if self.use_bias else None
        )

    def __call__(self, x):
        """
        Apply transposed convolution to the input tensor.
        
        Args:
            x: Input tensor of shape (batch, *spatial, in_channels).
        
        Returns:
            Output tensor of shape (batch, *spatial', out_channels).
        """
        n_spatial = len(self.kernel_size)
        stride = _as_tuple(self.stride, n_spatial)
        dilation = _as_tuple(self.dilation, n_spatial)
        padding = self.padding if isinstance(self.padding, str) else tuple(self.padding)

        y = jax.lax.conv_transpose(
            x, self.weight,
            strides=stride,
            padding=padding,
            rhs_dilation=dilation,
            dimension_numbers=_dimension_numbers(x.ndim),
        )
        if self.bias is not None:
            y = y + self.bias
        return y


__all__ = ["Conv", "ConvTranspose"]
