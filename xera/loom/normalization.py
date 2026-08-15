

from __future__ import annotations
import jax.numpy as jnp
from ..core import Module, Buffer, param
from .. import initializers


class LayerNorm(Module):
    
    dim: int
    eps: float = 1e-5

    def setup(self):
        self.gamma = param(self.rng(), initializers.ones(), (self.dim,))
        self.beta = param(self.rng(), initializers.zeros(), (self.dim,))

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        xn = (x - mean) / jnp.sqrt(var + self.eps)
        return xn * self.gamma + self.beta


class BatchNorm(Module):
    
    dim: int
    momentum: float = 0.9
    eps: float = 1e-5

    def setup(self):
        self.gamma = param(self.rng(), initializers.ones(), (self.dim,))
        self.beta = param(self.rng(), initializers.zeros(), (self.dim,))
        self.running_mean = Buffer(jnp.zeros(self.dim))
        self.running_var = Buffer(jnp.ones(self.dim))

    def __call__(self, x, *, training=True):
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
    
    new_bn = object.__new__(type(bn))
    new_bn.__dict__.update(bn.__dict__)
    new_bn.running_mean = Buffer(new_mean)
    new_bn.running_var = Buffer(new_var)
    return new_bn


__all__ = ["LayerNorm", "BatchNorm"]