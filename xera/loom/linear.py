

from __future__ import annotations
import jax.numpy as jnp
from ..core import Module, param
from .. import initializers


class Dense(Module):
    
    in_features: int
    out_features: int
    use_bias: bool = True

    def setup(self):
        self.weight = param(
            self.rng(), initializers.lecun_normal(),
            (self.in_features, self.out_features),
        )
        self.bias = (
            param(self.rng(), initializers.zeros(), (self.out_features,))
            if self.use_bias else None
        )

    def __call__(self, x):
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


__all__ = ["Dense"]