"""
Functional API: activations + attention + loss.

Mirrors `jax.nn`'s shape as a single namespace that mixes thin aliases
(`activations.py`, re-exporting `jax.nn`'s activation functions) with
original implementations (`attention.py`, re-exporting `flash_sdpa`,
and `loss.py`, providing the `*_loss` functions) -- same as `jax.nn`
mixes `jax.nn.relu` (alias) with `jax.nn.dot_product_attention`
(original implementation).

This is a top-level package (`xera.functional`), separate from `xera.loom`
-- layers live in `loom`, plain functional ops live here. The loss
functions used to live in the old `xera.weave` package as static
methods on a `Loss` class; each is a pure `(pred, target, ...) ->
scalar` computation with no state to carry, so they're plain top-level
functions here instead -- same treatment as the activations.

Example:
    >>> from xera.functional import relu, flash_sdpa, mae_loss, ce_loss
    >>> x = relu(x)
    >>> out = flash_sdpa(q, k, v, causal=True)
    >>> loss = ce_loss(logits, labels)
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
from .loss import (
    mae_loss,
    mse_loss,
    ce_loss,
    bce_loss,
    bce_with_logits_loss,
    hinge_loss,
    huber_loss,
    smooth_l1_loss,
    kl_div_loss,
    nll_loss,
    focal_loss,
    cosine_embedding_loss,
    margin_ranking_loss,
    rmse_loss,
    poisson_loss,
    gamma_loss,
    log_cosh_loss,
    quantile_loss,
    sigmoid_focal_cross_entropy_loss,
    triplet_loss,
    contrastive_loss,
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
    "flash_sdpa",
    "mae_loss",
    "mse_loss",
    "ce_loss",
    "bce_loss",
    "bce_with_logits_loss",
    "hinge_loss",
    "huber_loss",
    "smooth_l1_loss",
    "kl_div_loss",
    "nll_loss",
    "focal_loss",
    "cosine_embedding_loss",
    "margin_ranking_loss",
    "rmse_loss",
    "poisson_loss",
    "gamma_loss",
    "log_cosh_loss",
    "quantile_loss",
    "sigmoid_focal_cross_entropy_loss",
    "triplet_loss",
    "contrastive_loss",
]
