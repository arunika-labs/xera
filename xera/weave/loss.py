

"""
Loss functions module for JAX-based neural network training.

This module provides a comprehensive collection of loss functions commonly
used in machine learning, including regression losses, classification losses,
and specialized losses for specific tasks.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp


class Loss:
    """
    A collection of commonly used loss functions for neural network training.
    
    This class provides static methods for computing various loss functions
    in a JAX-compatible way. All methods are static and can be called directly
    on the class without instantiation.
    
    Example:
        >>> loss = Loss.L1(predictions, targets)
        >>> ce_loss = Loss.CE(logits, labels)
        >>> focal_loss = Loss.FocalLoss(logits, labels, alpha=0.25, gamma=2.0)
    """

    @staticmethod
    def L1(pred, target):
        """
        Compute L1 (Mean Absolute Error) loss.
        
        Args:
            pred: Predictions.
            target: Ground truth targets.
        
        Returns:
            Mean absolute error between predictions and targets.
        """
        return jnp.mean(jnp.abs(pred - target))

    @staticmethod
    def L2(pred, target):
        """
        Compute L2 (Mean Squared Error) loss.
        
        Args:
            pred: Predictions.
            target: Ground truth targets.
        
        Returns:
            Mean squared error between predictions and targets.
        """
        return jnp.mean(jnp.square(pred - target))

    @staticmethod
    def CE(logits, labels, axis=-1):
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

    @staticmethod
    def BCE(logits, labels):
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

    @staticmethod
    def BCEWithLogits(logits, labels):
        """
        Compute Binary Cross-Entropy with logits (same as BCE).
        
        This is an alias for BCE for compatibility with other frameworks.
        
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

    @staticmethod
    def Hinge(pred, target, margin=1.0):
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

    @staticmethod
    def Huber(pred, target, delta=1.0):
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

    @staticmethod
    def SmoothL1(pred, target, beta=1.0):
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

    @staticmethod
    def KLDiv(log_probs, target_probs, axis=-1):
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

    @staticmethod
    def NLL(log_probs, labels, axis=-1):
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

    @staticmethod
    def FocalLoss(logits, labels, alpha=0.25, gamma=2.0, axis=-1):
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

    @staticmethod
    def CosineEmbedding(pred1, pred2, target, margin=0.0):
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

    @staticmethod
    def MarginRanking(pred1, pred2, target, margin=1.0):
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

    @staticmethod
    def RMSE(pred, target):
        """
        Compute Root Mean Squared Error loss.
        
        Args:
            pred: Predictions.
            target: Ground truth targets.
        
        Returns:
            Root mean squared error.
        """
        return jnp.sqrt(jnp.mean(jnp.square(pred - target)))

    @staticmethod
    def Poisson(pred, target):
        """
        Compute Poisson loss for count data.
        
        Args:
            pred: Predicted counts (must be positive).
            target: Ground truth counts.
        
        Returns:
            Poisson loss.
        """
        return jnp.mean(pred - target * jnp.log(pred + 1e-8))

    @staticmethod
    def Gamma(pred, target):
        """
        Compute Gamma loss for positive continuous data.
        
        Args:
            pred: Predictions (must be positive).
            target: Ground truth targets (must be positive).
        
        Returns:
            Gamma loss.
        """
        return jnp.mean(jnp.log(pred + 1e-8) + target / (pred + 1e-8))

    @staticmethod
    def LogCosh(pred, target):
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

    @staticmethod
    def Quantile(pred, target, quantile=0.5):
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

    @staticmethod
    def SigmoidFocalCrossEntropy(logits, labels, alpha=0.25, gamma=2.0):
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

    @staticmethod
    def TripletLoss(anchor, positive, negative, margin=1.0):
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

    @staticmethod
    def ContrastiveLoss(pred1, pred2, target, margin=1.0):
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


__all__ = ["Loss"]
