"""
Functional API: activations + attention.

Mirrors `jax.nn`'s shape as a single namespace that mixes thin aliases
(`activations.py`, re-exporting `jax.nn`'s activation functions) with
original implementations (`attention.py`, re-exporting `flash_sdpa`) --
same as `jax.nn` mixes `jax.nn.relu` (alias) with
`jax.nn.dot_product_attention` (original implementation).

This is a top-level package (`xera.functional`), separate from `xera.loom`
-- layers live in `loom`, plain functional ops live here.

Example:
    >>> from xera.functional import relu, flash_sdpa
    >>> x = relu(x)
    >>> out = flash_sdpa(q, k, v, causal=True)
"""

from __future__ import annotations

from .activations import (
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
from .attention import flash_sdpa

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
    "flash_sdpa",
]
