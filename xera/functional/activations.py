"""
Functional activation and utility functions.

This module re-exports the standard set of activation functions from
`jax.nn` so they can be used directly from `xera.functional` without
requiring a separate `jax.nn` import. No reimplementation is done here
-- these are thin aliases over JAX's own (XLA-fused, well-tested)
implementations.

This is one module inside the `xera.functional` package. Unlike this
one, not everything under `functional/` is a thin alias -- see
`attention.py` in this same package, which holds an original
implementation (`auto_flash_attention`) rather than a re-export. The
package as a whole mirrors `jax.nn`, which is likewise a mix of thin
aliases (`jax.nn.relu`) and original implementations
(`jax.nn.dot_product_attention`).

Example:
    >>> from xera.functional import relu, gelu, silu
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
