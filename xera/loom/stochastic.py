

from __future__ import annotations
import jax
import jax.numpy as jnp
from ..core import Module


class Dropout(Module):
    
    rate: float

    def setup(self):
        pass  

    def __call__(self, x, *, key=None, deterministic=True):
        if deterministic or self.rate == 0.0:
            return x
        keep_prob = 1.0 - self.rate
        mask = jax.random.bernoulli(key, keep_prob, x.shape)
        return jnp.where(mask, x / keep_prob, 0.0)


__all__ = ["Dropout"]