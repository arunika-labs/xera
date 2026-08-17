

"""
State Space Models (SSM) and recurrent layers for sequence modeling.

This module provides implementations of State Space Models including S4D,
Selective State Space Models (SelectiveSSM), and the Mamba architecture.
These are efficient alternatives to transformers for long sequence modeling.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from ..core import Module, param
from .. import initializers
from .linear import Dense
from .conv import Conv


def _s4d_log_A_init(state_dim):
    """
    S4D (Structured State Space for Sequence Modeling Diagonal) initialization for A matrix.
    
    Initializes the log of the A matrix using a diagonal structure with
    values based on the state dimension indices. This follows the S4D
    initialization scheme for stable training.
    
    Args:
        state_dim: The state dimension for the SSM.
    
    Returns:
        An initialization function that creates log A values.
    """
    def init(key, shape, dtype=jnp.float32):
        channels, sd = shape
        n = jnp.arange(1, sd + 1, dtype=dtype)
        return jnp.broadcast_to(jnp.log(n), (channels, sd))
    return init


def _log_uniform_init(low, high):
    """
    Log-uniform initialization for parameters that should span multiple scales.
    
    Initializes parameters uniformly in log space, which is useful for
    parameters like time steps (dt) that should span several orders of magnitude.
    
    Args:
        low: Lower bound for the parameter (in linear space).
        high: Upper bound for the parameter (in linear space).
    
    Returns:
        An initialization function that samples from log-uniform distribution.
    """
    def init(key, shape, dtype=jnp.float32):
        u = jax.random.uniform(key, shape, dtype)
        return jnp.log(low) + u * (jnp.log(high) - jnp.log(low))
    return init


class SSM(Module):
    """
    Structured State Space Model (S4D variant).
    
    Implements a diagonal structured state space model with discretization
    using zero-order hold. This provides efficient sequence modeling with
    linear complexity in sequence length.
    
    Attributes:
        channels: Number of input/output channels.
        state_dim: Dimension of the state space (default: 16).
        dt_min: Minimum time step for discretization (default: 0.001).
        dt_max: Maximum time step for discretization (default: 0.1).
    
    Example:
        >>> ssm = SSM(channels=64, state_dim=16)
        >>> output = ssm(input_sequence)  # shape: (batch, seq_len, channels)
    """

    channels: int
    state_dim: int = 16
    dt_min: float = 0.001
    dt_max: float = 0.1

    def setup(self):
        """Initialize SSM parameters (A, B, C, D, dt)."""
        k_B, k_C, k_dt = self.rng(3)
        self.log_A = param(
            self.rng(), _s4d_log_A_init(self.state_dim), (self.channels, self.state_dim)
        )
        self.B = param(k_B, initializers.normal(stddev=1.0), (self.channels, self.state_dim))
        self.C = param(k_C, initializers.normal(stddev=1.0), (self.channels, self.state_dim))
        self.D = param(self.rng(), initializers.ones(), (self.channels,))
        self.log_dt = param(
            k_dt, _log_uniform_init(self.dt_min, self.dt_max), (self.channels,)
        )

    def __call__(self, x):
        """
        Apply the state space model to the input sequence.
        
        Args:
            x: Input tensor of shape (batch, seq_len, channels).
        
        Returns:
            Output tensor of shape (batch, seq_len, channels).
        """
        A = -jnp.exp(self.log_A)               # (channels, state_dim), always < 0
        dt = jnp.exp(self.log_dt)               # (channels,)
        dA = jnp.exp(A * dt[:, None])           # (channels, state_dim)
        dB = (dA - 1.0) / A * self.B            # exact ZOH discretization of B

        batch = x.shape[0]
        h0 = jnp.zeros((batch, self.channels, self.state_dim), dtype=x.dtype)
        xt = jnp.transpose(x, (1, 0, 2))        # (seq_len, batch, channels)

        def step(h, u_t):
            """Single step of the SSM recurrence."""
            h_new = dA[None] * h + dB[None] * u_t[:, :, None]
            y_t = jnp.sum(self.C[None] * h_new, axis=-1) + self.D[None] * u_t
            return h_new, y_t

        _, ys = jax.lax.scan(step, h0, xt)
        return jnp.transpose(ys, (1, 0, 2))     # (batch, seq_len, channels)


class SelectiveSSM(Module):
    """
    Selective State Space Model (SelectiveSSM).
    
    A variant of SSM where the discretization parameters (dt, B, C) are
    input-dependent, allowing the model to selectively remember or ignore
    information based on the input. This is the core building block of Mamba.
    
    Attributes:
        d_inner: Inner dimension of the SSM.
        state_dim: Dimension of the state space (default: 16).
        dt_rank: Rank for the low-rank dt projection (default: d_inner // 16).
    
    Example:
        >>> ssm = SelectiveSSM(d_inner=64, state_dim=16)
        >>> output = ssm(input_sequence)
    """

    d_inner: int
    state_dim: int = 16
    dt_rank: int = None

    def setup(self):
        """Initialize selective SSM parameters with input-dependent projections."""
        dt_rank = self.dt_rank if self.dt_rank is not None else max(1, self.d_inner // 16)
        self._dt_rank = dt_rank

        self.log_A = param(
            self.rng(), _s4d_log_A_init(self.state_dim), (self.d_inner, self.state_dim)
        )
        self.D = param(self.rng(), initializers.ones(), (self.d_inner,))

        # Projects each timestep's input to (dt_low, B, C) in one matmul --
        # dt_low is a low-rank bottleneck later expanded to d_inner by
        # dt_proj, matching the reference implementation's parameter budget
        # (a full d_inner -> d_inner dt projection per layer would be far
        # more parameters than the state itself justifies).
        self.x_proj = Dense(
            self.d_inner, dt_rank + 2 * self.state_dim, use_bias=False, key=self.rng()
        )
        self.dt_proj = Dense(dt_rank, self.d_inner, key=self.rng())

    def __call__(self, x):
        """
        Apply the selective state space model to the input sequence.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_inner).
        
        Returns:
            Output tensor of shape (batch, seq_len, d_inner).
        """
        A = -jnp.exp(self.log_A)  # (d_inner, state_dim), always < 0

        proj = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*state_dim)
        dt_low, B_seq, C_seq = jnp.split(
            proj, [self._dt_rank, self._dt_rank + self.state_dim], axis=-1
        )
        dt_seq = jax.nn.softplus(self.dt_proj(dt_low))  # (batch, seq_len, d_inner)

        batch, seq_len, _ = x.shape
        h0 = jnp.zeros((batch, self.d_inner, self.state_dim), dtype=x.dtype)

        xt = jnp.transpose(x, (1, 0, 2))          # (seq_len, batch, d_inner)
        dt_t = jnp.transpose(dt_seq, (1, 0, 2))    # (seq_len, batch, d_inner)
        B_t = jnp.transpose(B_seq, (1, 0, 2))      # (seq_len, batch, state_dim)
        C_t = jnp.transpose(C_seq, (1, 0, 2))      # (seq_len, batch, state_dim)

        def step(h, inputs):
            """Single step of the selective SSM recurrence."""
            u, dt, B, C = inputs
            dA = jnp.exp(A[None] * dt[:, :, None])       # (batch, d_inner, state_dim)
            dB = dt[:, :, None] * B[:, None, :]            # (batch, d_inner, state_dim)
            h_new = dA * h + dB * u[:, :, None]
            y = jnp.sum(C[:, None, :] * h_new, axis=-1) + self.D[None] * u
            return h_new, y

        _, ys = jax.lax.scan(step, h0, (xt, dt_t, B_t, C_t))
        return jnp.transpose(ys, (1, 0, 2))  # (batch, seq_len, d_inner)


class MambaBlock(Module):
    """
    Mamba block: A selective state space model architecture.
    
    Mamba is a linear-time sequence modeling architecture that combines
    selective state space models with gating mechanisms. It achieves
    transformer-quality performance while maintaining linear complexity
    in sequence length.
    
    Attributes:
        d_model: Model dimension (input/output dimension).
        d_inner: Inner dimension (default: d_model * 2).
        state_dim: State dimension for the SSM (default: 16).
        conv_kernel: Kernel size for the causal convolution (default: 4).
        dt_rank: Rank for the low-rank dt projection (default: d_inner // 16).
    
    Example:
        >>> mamba = MambaBlock(d_model=512, state_dim=16)
        >>> output = mamba(input_sequence)
    """

    d_model: int
    d_inner: int = None
    state_dim: int = 16
    conv_kernel: int = 4
    dt_rank: int = None

    def setup(self):
        """Initialize Mamba block components."""
        d_inner = self.d_inner if self.d_inner is not None else self.d_model * 2
        self._d_inner = d_inner

        self.in_proj = Dense(self.d_model, d_inner * 2, use_bias=False, key=self.rng())
        self.conv = Conv(
            d_inner, d_inner, kernel_size=(self.conv_kernel,),
            padding=((self.conv_kernel - 1, 0),),  # causal: pad left only
            groups=d_inner, use_bias=True, key=self.rng(),
        )
        self.ssm = SelectiveSSM(d_inner, state_dim=self.state_dim, dt_rank=self.dt_rank, key=self.rng())
        self.out_proj = Dense(d_inner, self.d_model, use_bias=False, key=self.rng())

    def __call__(self, x):
        """
        Apply the Mamba block to the input sequence.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        x_in, z = jnp.split(self.in_proj(x), 2, axis=-1)   # each (batch, seq_len, d_inner)
        x_in = jax.nn.silu(self.conv(x_in))
        y = self.ssm(x_in)
        y = y * jax.nn.silu(z)
        return self.out_proj(y)


__all__ = ["SSM", "SelectiveSSM", "MambaBlock"]
