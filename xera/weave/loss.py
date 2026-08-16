

from __future__ import annotations
import jax
import jax.numpy as jnp


class Loss:

    @staticmethod
    def L1(pred, target):
        return jnp.mean(jnp.abs(pred - target))

    @staticmethod
    def L2(pred, target):
        return jnp.mean(jnp.square(pred - target))

    @staticmethod
    def CE(logits, labels, axis=-1):
        log_probs = jax.nn.log_softmax(logits, axis=axis)
        if labels.ndim == logits.ndim:
            onehot = labels
        else:
            onehot = jax.nn.one_hot(labels, logits.shape[axis])
        return -jnp.mean(jnp.sum(onehot * log_probs, axis=axis))


__all__ = ["Loss"]
