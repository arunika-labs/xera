"""
Functional activation and utility functions.

This module re-exports the standard set of activation functions from
`jax.nn` so they can be used directly from `xera.loom` without requiring
a separate `jax.nn` import. No reimplementation is done here — these are
thin aliases over JAX's own (XLA-fused, well-tested) implementations.

Example:
    >>> from xera.loom import relu, gelu, silu
    >>> x = relu(x)
"""

from __future__ import annotations

from jax.nn import (
    celu,
    elu,
    gelu,
    glu,
    hard_sigmoid,
    hard_silu,
    hard_swish,
    hard_tanh,
    leaky_relu,
    log_sigmoid,
    log_softmax,
    logsumexp,
    mish,
    one_hot,
    relu,
    relu6,
    selu,
    sigmoid,
    silu,
    soft_sign,
    softmax,
    softplus,
    squareplus,
    standardize,
    swish,
    tanh,
)

__all__ = [
    "celu",
    "elu",
    "gelu",
    "glu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "logsumexp",
    "mish",
    "one_hot",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "soft_sign",
    "softmax",
    "softplus",
    "squareplus",
    "standardize",
    "swish",
    "tanh",
]
