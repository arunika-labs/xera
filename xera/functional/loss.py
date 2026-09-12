

"""
Loss functions module for JAX-based neural network training.

This module provides a comprehensive collection of loss functions commonly
used in machine learning, including regression losses, classification losses,
and specialized losses for specific tasks.

Lives under `xera.functional` (moved from the old `xera.weave` package)
alongside activations and attention. Unlike optimizers -- which carry
state across steps and so benefit from a class hierarchy -- a loss is
just a pure `(pred, target, ...) -> scalar` computation, so each one is
a plain top-level function here rather than a method on a class. This
mirrors how `activations.py` exposes `relu`, `gelu`, etc. as bare
functions instead of bundling them under an `Activations` class.

Example:
    >>> from xera.functional import mae_loss, ce_loss, focal_loss
    >>> loss = mae_loss(predictions, targets)
    >>> ce = ce_loss(logits, labels)
    >>> focal = focal_loss(logits, labels, alpha=0.25, gamma=2.0)
"""

from __future__ import annotations
import jax
import jax.numpy as jnp


def mae_loss(pred, target):
    """
    Compute L1 / Mean Absolute Error (MAE) loss.

    Args:
        pred: Predictions.
        target: Ground truth targets.

    Returns:
        Mean absolute error between predictions and targets.
    """
    return jnp.mean(jnp.abs(pred - target))


def mse_loss(pred, target):
    """
    Compute L2 / Mean Squared Error (MSE) loss.

    Args:
        pred: Predictions.
        target: Ground truth targets.

    Returns:
        Mean squared error between predictions and targets.
    """
    return jnp.mean(jnp.square(pred - target))


def ce_loss(logits, labels, axis=-1):
    """
    Compute Cross-Entropy loss for classification.

    Args:
        logits: Unnormalized log probabilities (logits).
        labels: Ground truth labels. Can be integer class indices or
            one-hot encoded vectors.
        axis: The axis containing the class probabilities (default: -1).

    Returns:
        Cross-entropy loss.
    """
    log_probs = jax.nn.log_softmax(logits, axis=axis)
    if labels.ndim == logits.ndim:
        onehot = labels
    else:
        onehot = jax.nn.one_hot(labels, logits.shape[axis])
    return -jnp.mean(jnp.sum(onehot * log_probs, axis=axis))


def bce_loss(logits, labels):
    """
    Compute Binary Cross-Entropy loss.

    Args:
        logits: Unnormalized log probabilities (logits).
        labels: Binary targets (0 or 1).

    Returns:
        Binary cross-entropy loss.
    """
    return -jnp.mean(
        labels * jax.nn.log_sigmoid(logits) +
        (1 - labels) * jax.nn.log_sigmoid(-logits)
    )


def bce_with_logits_loss(logits, labels):
    """
    Compute Binary Cross-Entropy with logits (same as `bce_loss`).

    This is an alias for `bce_loss` for compatibility with other
    frameworks that name it this way.

    Args:
        logits: Unnormalized log probabilities (logits).
        labels: Binary targets (0 or 1).

    Returns:
        Binary cross-entropy loss.
    """
    return bce_loss(logits, labels)


def hinge_loss(pred, target, margin=1.0):
    """
    Compute Hinge loss for SVM-style classification.

    Args:
        pred: Predictions.
        target: Ground truth labels (-1 or 1).
        margin: Margin parameter (default: 1.0).

    Returns:
        Hinge loss.
    """
    return jnp.mean(jnp.maximum(0, margin - target * pred))


def huber_loss(pred, target, delta=1.0):
    """
    Compute Huber loss, which is less sensitive to outliers than MSE.

    The Huber loss is quadratic for small errors and linear for large errors.

    Args:
        pred: Predictions.
        target: Ground truth targets.
        delta: Threshold where the loss changes from quadratic to linear.

    Returns:
        Huber loss.
    """
    error = pred - target
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return jnp.mean(0.5 * quadratic ** 2 + delta * linear)


def smooth_l1_loss(pred, target, beta=1.0):
    """
    Compute Smooth L1 loss (similar to Huber but with different parameterization).

    Args:
        pred: Predictions.
        target: Ground truth targets.
        beta: Threshold parameter (default: 1.0).

    Returns:
        Smooth L1 loss.
    """
    error = pred - target
    abs_error = jnp.abs(error)
    return jnp.where(
        abs_error < beta,
        0.5 * error ** 2 / beta,
        abs_error - 0.5 * beta
    ).mean()


def kl_div_loss(log_probs, target_probs, axis=-1):
    """
    Compute Kullback-Leibler divergence loss.

    Args:
        log_probs: Log probabilities from the model.
        target_probs: Target probability distribution.
        axis: The axis containing the class probabilities (default: -1).

    Returns:
        KL divergence loss.
    """
    return jnp.sum(target_probs * (jnp.log(target_probs) - log_probs), axis=axis).mean()


def nll_loss(log_probs, labels, axis=-1):
    """
    Compute Negative Log-Likelihood loss.

    Args:
        log_probs: Log probabilities from the model.
        labels: Ground truth labels. Can be integer class indices or
            one-hot encoded vectors.
        axis: The axis containing the class probabilities (default: -1).

    Returns:
        Negative log-likelihood loss.
    """
    if labels.ndim == log_probs.ndim:
        onehot = labels
    else:
        onehot = jax.nn.one_hot(labels, log_probs.shape[axis])
    return -jnp.mean(jnp.sum(onehot * log_probs, axis=axis))


def focal_loss(logits, labels, alpha=0.25, gamma=2.0, axis=-1):
    """
    Compute Focal Loss for addressing class imbalance.

    Focal loss down-weights well-classified examples to focus training
    on hard examples.

    Args:
        logits: Unnormalized log probabilities (logits).
        labels: Ground truth labels. Can be integer class indices or
            one-hot encoded vectors.
        alpha: Weighting factor for rare class (default: 0.25).
        gamma: Focusing parameter (default: 2.0).
        axis: The axis containing the class probabilities (default: -1).

    Returns:
        Focal loss.
    """
    probs = jax.nn.softmax(logits, axis=axis)
    if labels.ndim == logits.ndim:
        onehot = labels
    else:
        onehot = jax.nn.one_hot(labels, logits.shape[axis])

    pt = jnp.sum(onehot * probs, axis=axis)
    focal_weight = (1 - pt) ** gamma
    alpha_weight = alpha * onehot + (1 - alpha) * (1 - onehot)

    log_probs = jax.nn.log_softmax(logits, axis=axis)
    loss = -alpha_weight * focal_weight * onehot * log_probs
    return jnp.sum(loss, axis=axis).mean()


def cosine_embedding_loss(pred1, pred2, target, margin=0.0):
    """
    Compute Cosine Embedding loss for metric learning.

    Args:
        pred1: First set of predictions.
        pred2: Second set of predictions.
        target: Target labels (1 for similar, -1 for dissimilar).
        margin: Margin for dissimilar pairs (default: 0.0).

    Returns:
        Cosine embedding loss.
    """
    cosine = jnp.sum(pred1 * pred2, axis=-1) / (
        jnp.linalg.norm(pred1, axis=-1) * jnp.linalg.norm(pred2, axis=-1) + 1e-8
    )
    return jnp.where(
        target == 1,
        1 - cosine,
        jnp.maximum(0, cosine - margin)
    ).mean()


def margin_ranking_loss(pred1, pred2, target, margin=1.0):
    """
    Compute Margin Ranking loss for learning to rank.

    Args:
        pred1: First set of predictions.
        pred2: Second set of predictions.
        target: Target labels (1 if pred1 should rank higher, -1 otherwise).
        margin: Margin parameter (default: 1.0).

    Returns:
        Margin ranking loss.
    """
    return jnp.mean(jnp.maximum(0, margin - target * (pred1 - pred2)))


def rmse_loss(pred, target):
    """
    Compute Root Mean Squared Error loss.

    Args:
        pred: Predictions.
        target: Ground truth targets.

    Returns:
        Root mean squared error.
    """
    return jnp.sqrt(jnp.mean(jnp.square(pred - target)))


def poisson_loss(pred, target):
    """
    Compute Poisson loss for count data.

    Args:
        pred: Predicted counts (must be positive).
        target: Ground truth counts.

    Returns:
        Poisson loss.
    """
    return jnp.mean(pred - target * jnp.log(pred + 1e-8))


def gamma_loss(pred, target):
    """
    Compute Gamma loss for positive continuous data.

    Args:
        pred: Predictions (must be positive).
        target: Ground truth targets (must be positive).

    Returns:
        Gamma loss.
    """
    return jnp.mean(jnp.log(pred + 1e-8) + target / (pred + 1e-8))


def log_cosh_loss(pred, target):
    """
    Compute Logarithm of Hyperbolic Cosine loss.

    This loss is similar to Huber loss but is twice differentiable everywhere.

    Args:
        pred: Predictions.
        target: Ground truth targets.

    Returns:
        Log cosh loss.
    """
    error = pred - target
    return jnp.mean(jnp.log(jnp.cosh(error)))


def quantile_loss(pred, target, quantile=0.5):
    """
    Compute Quantile loss for quantile regression.

    Args:
        pred: Predictions.
        target: Ground truth targets.
        quantile: Target quantile (default: 0.5 for median).

    Returns:
        Quantile loss.
    """
    error = pred - target
    return jnp.mean(jnp.maximum(quantile * error, (quantile - 1) * error))


def sigmoid_focal_cross_entropy_loss(logits, labels, alpha=0.25, gamma=2.0):
    """
    Compute Sigmoid Focal Cross-Entropy loss for binary classification.

    Combines sigmoid activation with focal loss for handling class imbalance
    in binary classification tasks.

    Args:
        logits: Unnormalized log probabilities (logits).
        labels: Binary targets (0 or 1).
        alpha: Weighting factor for positive class (default: 0.25).
        gamma: Focusing parameter (default: 2.0).

    Returns:
        Sigmoid focal cross-entropy loss.
    """
    probs = jax.nn.sigmoid(logits)
    if labels.ndim == logits.ndim:
        target = labels
    else:
        target = labels.astype(logits.dtype)

    pt = jnp.where(target == 1, probs, 1 - probs)
    alpha_weight = jnp.where(target == 1, alpha, 1 - alpha)
    focal_weight = (1 - pt) ** gamma

    bce = -jnp.log(pt + 1e-8)
    return jnp.mean(alpha_weight * focal_weight * bce)


def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet loss for metric learning.

    Encourages anchor-positive pairs to be closer than anchor-negative pairs
    by at least the specified margin.

    Args:
        anchor: Anchor embeddings.
        positive: Positive embeddings (same class as anchor).
        negative: Negative embeddings (different class from anchor).
        margin: Minimum margin between positive and negative distances.

    Returns:
        Triplet loss.
    """
    pos_dist = jnp.sum(jnp.square(anchor - positive), axis=-1)
    neg_dist = jnp.sum(jnp.square(anchor - negative), axis=-1)
    return jnp.mean(jnp.maximum(0, pos_dist - neg_dist + margin))


def contrastive_loss(pred1, pred2, target, margin=1.0):
    """
    Compute Contrastive loss for metric learning.

    Pulls similar pairs together and pushes dissimilar pairs apart.

    Args:
        pred1: First set of embeddings.
        pred2: Second set of embeddings.
        target: Target labels (1 for similar, 0 for dissimilar).
        margin: Margin for dissimilar pairs (default: 1.0).

    Returns:
        Contrastive loss.
    """
    dist = jnp.sum(jnp.square(pred1 - pred2), axis=-1)
    return jnp.mean(
        target * dist +
        (1 - target) * jnp.maximum(0, margin - dist)
    )


__all__ = [
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
