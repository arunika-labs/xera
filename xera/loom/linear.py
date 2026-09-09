"""
Linear (dense) layers for fully-connected neural network operations.

This module provides the standard dense/fully-connected layer that applies
a linear transformation to input data, optionally followed by a bias addition,
along with low-rank adapter variants (LoRA, DoRA) commonly used for
parameter-efficient fine-tuning of large pretrained weights.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from .module import Module, param
from . import initializers


class Linear(Module):
    """
    Standard dense (fully-connected) layer.
    
    Applies a linear transformation to the input: y = x @ W + b
    where W is the weight matrix and b is the optional bias vector.
    This is the fundamental building block for most neural networks.
    
    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        use_bias: Whether to add a bias term (default: True).
    
    Example:
        >>> layer = Linear(in_features=128, out_features=64)
        >>> output = layer(input_tensor)  # shape: (..., 64)
    """
    
    in_features: int
    out_features: int
    use_bias: bool = True

    def setup(self):
        """Initialize weight matrix and optional bias vector."""
        self.weight = param(
            self.rng(), initializers.lecun_normal(),
            (self.in_features, self.out_features),
        )
        self.bias = (
            param(self.rng(), initializers.zeros(), (self.out_features,))
            if self.use_bias else None
        )

    def __call__(self, x):
        """
        Apply the linear transformation to the input.
        
        Args:
            x: Input tensor of shape (..., in_features).
        
        Returns:
            Output tensor of shape (..., out_features).
        """
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class LoRALinear(Module):
    """
    Linear layer adapted with LoRA (Low-Rank Adaptation).

    Wraps a frozen base weight matrix with a trainable low-rank update:

        y = x @ (W_base + B @ A) + b

    where `W_base` is a pretrained weight (kept fixed via `stop_gradient`
    every forward pass -- it never receives gradient updates through this
    layer, regardless of whether it happens to sit in a `jax.grad` call
    that also touches other parameters), and `A` (in_features x rank),
    `B` (rank x out_features) are the trainable low-rank factors from
    "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al.,
    2021, https://arxiv.org/abs/2106.09685).

    `B` is initialized to zero, so the adapted layer computes exactly
    `x @ W_base + b` at initialization -- the low-rank update starts as a
    true no-op and only diverges from the frozen base as training
    proceeds.

    For the common case of adapting a small number of large pretrained
    weights, `base_weight` is supplied once at construction and owned by
    this layer (frozen). If instead a base weight needs to be shared
    across many adapters at once -- e.g. many low-rank "virtual experts"
    built on top of one common base, as in mixture-of-experts-style
    architectures -- pass the base weight as an argument to `__call__`
    instead of at construction; see `__call__`'s `base_weight` parameter.

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        rank: Rank of the low-rank update (typically << min(in_features,
            out_features); common values are 4-64).
        base_weight: Optional frozen base weight, shape (in_features,
            out_features). If provided here, this layer owns and freezes
            it, and it must NOT also be passed to `__call__`. If omitted,
            it must be supplied externally on every call instead (shared
            base case).
        use_bias: Whether to add a trainable bias term (default: False --
            LoRA conventionally leaves bias untouched or omits it, since
            the adaptation targets weight matrices).

    Example:
        >>> # Adapting an owned, frozen pretrained weight:
        >>> pretrained_w = jnp.load("layer_weight.npy")
        >>> layer = LoRALinear(768, 768, rank=8, base_weight=pretrained_w, key=key)
        >>> output = layer(input_tensor)
        >>>
        >>> # Sharing one base weight across many low-rank adapters:
        >>> shared_base = jnp.zeros((768, 3072))
        >>> experts = [LoRALinear(768, 3072, rank=8, key=k) for k in keys]
        >>> outputs = [e(x, base_weight=shared_base) for e in experts]
    """

    in_features: int
    out_features: int
    rank: int
    base_weight: jnp.ndarray = None
    use_bias: bool = False

    def setup(self):
        """Initialize low-rank factors A, B and optional bias/owned base."""
        self.lora_a = param(
            self.rng(), initializers.kaiming_uniform(),
            (self.in_features, self.rank),
        )
        # B starts at zero so the adapter is a no-op at initialization --
        # the effective weight equals exactly base_weight until training
        # moves lora_a/lora_b away from this starting point.
        self.lora_b = param(
            self.rng(), initializers.zeros(),
            (self.rank, self.out_features),
        )
        self.bias = (
            param(self.rng(), initializers.zeros(), (self.out_features,))
            if self.use_bias else None
        )

    def __call__(self, x, base_weight=None):
        """
        Apply the LoRA-adapted linear transformation.

        Args:
            x: Input tensor of shape (..., in_features).
            base_weight: Base weight to adapt, shape (in_features,
                out_features). Required if this layer was constructed
                without `base_weight` (shared-base case); must be omitted
                if it was (owned-base case) -- passing both, or neither,
                raises `ValueError`.

        Returns:
            Output tensor of shape (..., out_features).

        Raises:
            ValueError: If `base_weight` is ambiguous -- neither an owned
                base nor an argument was provided, or both were.
        """
        w_base = self._resolve_base(base_weight)
        w_eff = jax.lax.stop_gradient(w_base) + self.lora_a @ self.lora_b
        y = x @ w_eff
        if self.bias is not None:
            y = y + self.bias
        return y

    def _resolve_base(self, base_weight_arg):
        """Pick the owned base_weight or the one passed to `__call__`."""
        owned = self.base_weight is not None
        passed = base_weight_arg is not None
        if owned == passed:
            raise ValueError(
                "LoRALinear: provide base_weight either at construction "
                "(owned, frozen base) or to __call__ (shared base), not "
                "both and not neither."
            )
        return self.base_weight if owned else base_weight_arg


class DoRALinear(Module):
    """
    Linear layer adapted with DoRA (Weight-Decomposed Low-Rank Adaptation).

    DoRA ("DoRA: Weight-Decomposed Low-Rank Adaptation", Liu et al., 2024,
    https://arxiv.org/abs/2402.09353) extends LoRA by decomposing the
    adapted weight into magnitude and direction:

        W' = W_base + B @ A
        y  = x @ ( m * (W' / ||W'||_c) ) + b

    where `||W'||_c` is the column-wise (output-dimension) L2 norm of
    `W'`, and `m` is a trainable per-output-column magnitude vector.
    LoRA's `A`/`B` only ever shape the *direction* `W'` moves in; DoRA
    additionally lets each output column's *magnitude* be learned
    independently of that direction, which the DoRA paper shows tracks
    full fine-tuning's learning behavior more closely than plain LoRA at
    matched rank.

    As in `LoRALinear`, `base_weight` is frozen via `stop_gradient` every
    forward pass, and `B` is zero-initialized so the adapter starts as a
    near-no-op (the magnitude vector `m` is initialized from the base
    weight's own column norms, so the *initial* effective weight equals
    `W_base` exactly -- see `setup`).

    Ownership of `base_weight` follows the same two modes as
    `LoRALinear`: supply it at construction to have this layer own and
    freeze it, or omit it there and pass it to `__call__` instead to
    share one base weight across many DoRA adapters (e.g. many low-rank
    "virtual experts" built on a common base FFN or attention
    projection).

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        rank: Rank of the low-rank update (typically << min(in_features,
            out_features); common values are 4-64).
        base_weight: Optional frozen base weight, shape (in_features,
            out_features). If provided here, this layer owns and freezes
            it, and it must NOT also be passed to `__call__`. If omitted,
            it must be supplied externally on every call instead (shared
            base case).
        use_bias: Whether to add a trainable bias term (default: False).

    Example:
        >>> # Adapting an owned, frozen pretrained weight:
        >>> pretrained_w = jnp.load("layer_weight.npy")
        >>> layer = DoRALinear(768, 768, rank=8, base_weight=pretrained_w, key=key)
        >>> output = layer(input_tensor)
        >>>
        >>> # Sharing one base weight across many DoRA "virtual experts":
        >>> shared_base_up = jnp.zeros((768, 3072))
        >>> experts = [DoRALinear(768, 3072, rank=16, key=k) for k in keys]
        >>> expert_outputs = [e(x, base_weight=shared_base_up) for e in experts]
        >>> combined = sum(expert_outputs)  # additive combination, e.g. MoE-style
    """

    in_features: int
    out_features: int
    rank: int
    base_weight: jnp.ndarray = None
    use_bias: bool = False

    def setup(self):
        """Initialize low-rank factors, magnitude vector, and optional bias/owned base."""
        self.lora_a = param(
            self.rng(), initializers.kaiming_uniform(),
            (self.in_features, self.rank),
        )
        self.lora_b = param(
            self.rng(), initializers.zeros(),
            (self.rank, self.out_features),
        )
        if self.base_weight is not None:
            # Owned base: initialize magnitude from its own column norms,
            # so the effective weight at init equals base_weight exactly
            # (lora_b is zero, so W' = base_weight, and m = ||base_weight||_c
            # cancels the normalization).
            init_magnitude = jnp.linalg.norm(self.base_weight, axis=0)
        else:
            # Shared base supplied only at call time -- fall back to a
            # plain-ones start; the first call's column norms won't be
            # known until then, so exact equivalence to base_weight at
            # init isn't guaranteed in the shared-base case.
            init_magnitude = jnp.ones((self.out_features,))
        self.magnitude = param(self.rng(), lambda *_: init_magnitude, (self.out_features,))
        self.bias = (
            param(self.rng(), initializers.zeros(), (self.out_features,))
            if self.use_bias else None
        )

    def __call__(self, x, base_weight=None):
        """
        Apply the DoRA-adapted linear transformation.

        Args:
            x: Input tensor of shape (..., in_features).
            base_weight: Base weight to adapt, shape (in_features,
                out_features). Required if this layer was constructed
                without `base_weight` (shared-base case); must be omitted
                if it was (owned-base case) -- passing both, or neither,
                raises `ValueError`.

        Returns:
            Output tensor of shape (..., out_features).

        Raises:
            ValueError: If `base_weight` is ambiguous -- neither an owned
                base nor an argument was provided, or both were.
        """
        w_base = self._resolve_base(base_weight)
        w_prime = jax.lax.stop_gradient(w_base) + self.lora_a @ self.lora_b
        col_norm = jnp.linalg.norm(w_prime, axis=0, keepdims=True) + 1e-6
        w_eff = self.magnitude[None, :] * (w_prime / col_norm)
        y = x @ w_eff
        if self.bias is not None:
            y = y + self.bias
        return y

    def _resolve_base(self, base_weight_arg):
        """Pick the owned base_weight or the one passed to `__call__`."""
        owned = self.base_weight is not None
        passed = base_weight_arg is not None
        if owned == passed:
            raise ValueError(
                "DoRALinear: provide base_weight either at construction "
                "(owned, frozen base) or to __call__ (shared base), not "
                "both and not neither."
            )
        return self.base_weight if owned else base_weight_arg


__all__ = ["Linear", "LoRALinear", "DoRALinear"]
