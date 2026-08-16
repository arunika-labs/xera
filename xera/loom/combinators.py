

from __future__ import annotations
from typing import Any
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


class Residual(Module):
    """Wraps a module (or any callable Module) as `x + inner(x)`. `inner`
    must preserve `x`'s shape.

    Any `**kwargs` passed to this wrapper's `__call__` are forwarded to
    `inner` -- but note `Sequential` above only forwards kwargs to layers
    it recognizes as `Dropout` (a pre-existing limitation, not something
    this class works around). If `inner` contains a `Dropout` and needs
    `key=`/`deterministic=` threaded through, call `Residual` directly
    rather than through `Sequential`, or rely on `Dropout`'s defaults
    (`deterministic=True`) when used inside one.
    """
    inner: Any

    def setup(self):
        pass

    def __call__(self, x, **kwargs):
        return x + self.inner(x, **kwargs)


class Lambda(Module):
    """Wraps a plain function as a Module, so it can slot in anywhere a
    layer is expected (e.g. inside `Sequential`). `fn` is stored as static
    config, not a pytree leaf -- fine for stateless functions like
    `jax.nn.gelu` or a reshape; if you need learnable params, write a
    proper Module instead of trying to close over them here.
    """
    fn: Any

    def setup(self):
        pass

    def __call__(self, x, **kwargs):
        return self.fn(x)


__all__ = ["Sequential", "Residual", "Lambda"]
