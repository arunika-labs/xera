"""
Normalization layers for stabilizing and accelerating neural network training.

This module provides various normalization techniques including layer normalization,
batch normalization, group normalization, and their variants with running statistics.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from .module import Module, Buffer, param
from . import initializers


class LayerNorm(Module):
    """
    Layer Normalization layer.
    
    Normalizes the activations of a layer across the feature dimension.
    Unlike batch normalization, layer normalization doesn't depend on the
    batch statistics and is commonly used in transformer architectures.
    
    Attributes:
        dim: The feature dimension to normalize over.
        eps: Small constant for numerical stability (default: 1e-5).
    
    Example:
        >>> ln = LayerNorm(dim=512)
        >>> output = ln(input_tensor)
    """
    
    dim: int
    eps: float = 1e-5

    def setup(self):
        """Initialize scale (gamma) and shift (beta) parameters."""
        self.gamma = param(self.rng(), initializers.ones(), (self.dim,))
        self.beta = param(self.rng(), initializers.zeros(), (self.dim,))

    def __call__(self, x):
        """
        Apply layer normalization to the input.
        
        Args:
            x: Input tensor of shape (..., dim).
        
        Returns:
            Normalized tensor of the same shape.
        """
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        xn = (x - mean) / jnp.sqrt(var + self.eps)
        return xn * self.gamma + self.beta


class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization.
    
    A simplified variant of layer normalization that removes the mean centering
    operation, keeping only the RMS scaling. This is commonly used in large
    language models and can be more computationally efficient.
    
    Attributes:
        dim: The feature dimension to normalize over.
        eps: Small constant for numerical stability (default: 1e-6).
    
    Example:
        >>> rms = RMSNorm(dim=512)
        >>> output = rms(input_tensor)
    """

    dim: int
    eps: float = 1e-6

    def setup(self):
        """Initialize scale (gamma) parameter."""
        self.gamma = param(self.rng(), initializers.ones(), (self.dim,))

    def __call__(self, x):
        """
        Apply RMS normalization to the input.
        
        Args:
            x: Input tensor of shape (..., dim).
        
        Returns:
            Normalized tensor of the same shape.
        """
        ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        xn = x * jax.lax.rsqrt(ms + self.eps)
        return xn * self.gamma


class BatchNorm(Module):
    """
    Batch Normalization layer with running statistics.
    
    Normalizes activations across the batch dimension, maintaining running
    statistics for use during inference. This helps reduce internal covariate
    shift and can significantly accelerate training.
    
    Attributes:
        dim: The feature dimension to normalize over.
        momentum: Momentum for updating running statistics (default: 0.9).
        eps: Small constant for numerical stability (default: 1e-5).
    
    Example:
        >>> bn = BatchNorm(dim=64)
        >>> output, new_bn = bn(input_tensor, deterministic=False)
    """
    
    dim: int
    momentum: float = 0.9
    eps: float = 1e-5

    def setup(self):
        """Initialize parameters and running statistics buffers."""
        self.gamma = param(self.rng(), initializers.ones(), (self.dim,))
        self.beta = param(self.rng(), initializers.zeros(), (self.dim,))
        self.running_mean = Buffer(jnp.zeros(self.dim))
        self.running_var = Buffer(jnp.ones(self.dim))

    def __call__(self, x, *, deterministic=True):
        """
        Apply batch normalization to the input.
        
        Args:
            x: Input tensor of shape (batch, ..., dim).
            deterministic: If False, use batch statistics and update running
                stats (training mode). If True (default), use running
                statistics (inference/eval mode).
        
        Returns:
            A tuple (output, new_layer) where new_layer contains updated
            running statistics if deterministic=False.
        """
        training = not deterministic
        if training:
            mean = jnp.mean(x, axis=0)
            var = jnp.var(x, axis=0)
            new_running_mean = self.momentum * self.running_mean.value + (1 - self.momentum) * mean
            new_running_var = self.momentum * self.running_var.value + (1 - self.momentum) * var
            
            
            new_self = _replace_state(self, new_running_mean, new_running_var)
            xn = (x - mean) / jnp.sqrt(var + self.eps)
            return xn * self.gamma + self.beta, new_self
        else:
            xn = (x - self.running_mean.value) / jnp.sqrt(self.running_var.value + self.eps)
            return xn * self.gamma + self.beta, self


def _replace_state(bn, new_mean, new_var):
    """
    Create a new BatchNorm instance with updated running statistics.
    
    Args:
        bn: The original BatchNorm layer.
        new_mean: New running mean values.
        new_var: New running variance values.
    
    Returns:
        A new BatchNorm instance with updated running statistics.
    """
    new_bn = object.__new__(type(bn))
    new_bn.__dict__.update(bn.__dict__)
    new_bn.running_mean = Buffer(new_mean)
    new_bn.running_var = Buffer(new_var)
    return new_bn


class GroupNorm(Module):
    """
    Group Normalization layer.
    
    Divides channels into groups and normalizes within each group.
    This is an alternative to batch normalization that doesn't depend on
    batch size, making it suitable for small batches or transfer learning.
    
    Attributes:
        num_groups: Number of groups to divide channels into.
        dim: The feature dimension (must be divisible by num_groups).
        eps: Small constant for numerical stability (default: 1e-5).
    
    Example:
        >>> gn = GroupNorm(num_groups=8, dim=64)
        >>> output = gn(input_tensor)
    """
    
    num_groups: int
    dim: int
    eps: float = 1e-5

    def setup(self):
        """Initialize scale (gamma) and shift (beta) parameters."""
        self.gamma = param(self.rng(), initializers.ones(), (self.dim,))
        self.beta = param(self.rng(), initializers.zeros(), (self.dim,))

    def __call__(self, x):
        """
        Apply group normalization to the input.
        
        Args:
            x: Input tensor of shape (batch, *spatial, channels).
        
        Returns:
            Normalized tensor of the same shape.
        """
        batch, *spatial, channels = x.shape
        x_reshaped = x.reshape(batch, -1, self.num_groups, channels // self.num_groups)
        mean = jnp.mean(x_reshaped, axis=1, keepdims=True)
        var = jnp.var(x_reshaped, axis=1, keepdims=True)
        xn = (x_reshaped - mean) / jnp.sqrt(var + self.eps)
        xn = xn.reshape(x.shape)
        return xn * self.gamma + self.beta


class InstanceNorm(Module):
    """
    Instance Normalization layer.
    
    Normalizes each sample in the batch independently across spatial dimensions.
    Commonly used in style transfer and generative models where style
    information should be normalized per instance.
    
    Attributes:
        dim: The feature dimension to normalize over.
        eps: Small constant for numerical stability (default: 1e-5).
    
    Example:
        >>> inorm = InstanceNorm(dim=64)
        >>> output = inorm(input_tensor)
    """
    
    dim: int
    eps: float = 1e-5

    def setup(self):
        """Initialize scale (gamma) and shift (beta) parameters."""
        self.gamma = param(self.rng(), initializers.ones(), (self.dim,))
        self.beta = param(self.rng(), initializers.zeros(), (self.dim,))

    def __call__(self, x):
        """
        Apply instance normalization to the input.
        
        Args:
            x: Input tensor of shape (batch, *spatial, channels).
        
        Returns:
            Normalized tensor of the same shape.
        """
        spatial_dims = tuple(range(1, x.ndim - 1))
        mean = jnp.mean(x, axis=spatial_dims, keepdims=True)
        var = jnp.var(x, axis=spatial_dims, keepdims=True)
        xn = (x - mean) / jnp.sqrt(var + self.eps)
        return xn * self.gamma + self.beta


class LayerScale(Module):
    """
    Layer Scale for stabilizing training of deep residual networks.
    
    Applies a learnable scale factor to the input, commonly used in
    conjunction with residual connections to help with training
    very deep networks.
    
    Attributes:
        dim: The feature dimension.
        init_value: Initial value for the scale (default: 1e-5).
    
    Example:
        >>> ls = LayerScale(dim=256, init_value=1e-5)
        >>> output = ls(input_tensor)
    """
    
    dim: int
    init_value: float = 1e-5

    def setup(self):
        """Initialize the learnable scale parameter."""
        self.scale = param(
            self.rng(), 
            lambda key, shape, dtype: jnp.full(shape, self.init_value, dtype), 
            (self.dim,)
        )

    def __call__(self, x):
        """
        Apply layer scale to the input.
        
        Args:
            x: Input tensor of shape (..., dim).
        
        Returns:
            Scaled tensor of the same shape.
        """
        return self.scale * x


class GroupNormWithRunningStats(Module):
    """
    Group Normalization with running statistics.
    
    Combines group normalization with running statistics similar to batch
    normalization. This provides the benefits of group normalization while
    maintaining consistent behavior during training and inference.
    
    Attributes:
        num_groups: Number of groups to divide channels into.
        dim: The feature dimension (must be divisible by num_groups).
        momentum: Momentum for updating running statistics (default: 0.9).
        eps: Small constant for numerical stability (default: 1e-5).
    
    Example:
        >>> gn = GroupNormWithRunningStats(num_groups=8, dim=64)
        >>> output, new_gn = gn(input_tensor, deterministic=False)
    """
    
    num_groups: int
    dim: int
    momentum: float = 0.9
    eps: float = 1e-5

    def setup(self):
        """Initialize parameters and running statistics buffers."""
        self.gamma = param(self.rng(), initializers.ones(), (self.dim,))
        self.beta = param(self.rng(), initializers.zeros(), (self.dim,))
        self.running_mean = Buffer(jnp.zeros(self.dim))
        self.running_var = Buffer(jnp.ones(self.dim))

    def __call__(self, x, *, deterministic=True):
        """
        Apply group normalization with running statistics to the input.
        
        Args:
            x: Input tensor of shape (batch, *spatial, channels).
            deterministic: If False, use group statistics and update running
                stats (training mode). If True (default), use running
                statistics (inference/eval mode).
        
        Returns:
            A tuple (output, new_layer) where new_layer contains updated
            running statistics if deterministic=False.
        """
        batch, *spatial, channels = x.shape
        x_reshaped = x.reshape(batch, -1, self.num_groups, channels // self.num_groups)
        
        training = not deterministic
        if training:
            mean = jnp.mean(x_reshaped, axis=1, keepdims=True)
            var = jnp.var(x_reshaped, axis=1, keepdims=True)
            
            group_mean = mean.reshape(-1, channels)
            group_var = var.reshape(-1, channels)
            new_running_mean = self.momentum * self.running_mean.value + (1 - self.momentum) * group_mean.mean(axis=0)
            new_running_var = self.momentum * self.running_var.value + (1 - self.momentum) * group_var.mean(axis=0)
            
            new_self = _replace_group_state(self, new_running_mean, new_running_var)
            xn = (x_reshaped - mean) / jnp.sqrt(var + self.eps)
            xn = xn.reshape(x.shape)
            return xn * self.gamma + self.beta, new_self
        else:
            running_mean = self.running_mean.value.reshape(1, 1, self.num_groups, channels // self.num_groups)
            running_var = self.running_var.value.reshape(1, 1, self.num_groups, channels // self.num_groups)
            xn = (x_reshaped - running_mean) / jnp.sqrt(running_var + self.eps)
            xn = xn.reshape(x.shape)
            return xn * self.gamma + self.beta, self


def _replace_group_state(gn, new_mean, new_var):
    """
    Create a new GroupNormWithRunningStats instance with updated running statistics.
    
    Args:
        gn: The original GroupNormWithRunningStats layer.
        new_mean: New running mean values.
        new_var: New running variance values.
    
    Returns:
        A new GroupNormWithRunningStats instance with updated running statistics.
    """
    new_gn = object.__new__(type(gn))
    new_gn.__dict__.update(gn.__dict__)
    new_gn.running_mean = Buffer(new_mean)
    new_gn.running_var = Buffer(new_var)
    return new_gn


__all__ = [
    "LayerNorm", 
    "RMSNorm", 
    "BatchNorm", 
    "GroupNorm", 
    "InstanceNorm", 
    "LayerScale",
    "GroupNormWithRunningStats"
]
