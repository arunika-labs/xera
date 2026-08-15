

from __future__ import annotations
from ..core import Module
from .stochastic import Dropout


class Sequential(Module):
    
    layers: list

    def setup(self):
        pass

    def __call__(self, x, **kwargs):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer(x, **kwargs)
            else:
                x = layer(x)
        return x


__all__ = ["Sequential"]