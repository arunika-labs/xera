

from __future__ import annotations
import jax
import jax.numpy as jnp
from ..core import Module, param
from .. import initializers
from .linear import Dense


def _s4d_log_A_init(state_dim):
    """Deterministic S4D-Lin-style init: A_n = -(n+1) for n = 0..state_dim-1,
    stored in log space (since A itself is recovered as `-exp(log_A)`, always
    negative -- this parameterization keeps the recurrence stable under
    gradient updates, since no value of log_A can push A to be non-negative).
    Same spectrum for every channel; channels differentiate via B/C instead.
    Deterministic (ignores the RNG key) -- this is a fixed spectral
    initialization from the S4D paper, not a random one.
    """
    def init(key, shape, dtype=jnp.float32):
        channels, sd = shape
        n = jnp.arange(1, sd + 1, dtype=dtype)
        return jnp.broadcast_to(jnp.log(n), (channels, sd))
    return init


def _log_uniform_init(low, high):
    """Log-uniform between `low` and `high` -- standard for SSM timestep
    (dt) initialization, so a layer's channels span multiple timescales
    from the start rather than all starting at the same dt.
    """
    def init(key, shape, dtype=jnp.float32):
        u = jax.random.uniform(key, shape, dtype)
        return jnp.log(low) + u * (jnp.log(high) - jnp.log(low))
    return init


class SSM(Module):
    """Diagonal state-space layer, depthwise over channels (S4D-style,
    real-valued): each channel gets its own independent linear recurrence
    of size `state_dim`, with fixed (non-input-dependent) dynamics,
    discretized via zero-order hold.

    Simplified to real-valued A/B/C rather than the complex-valued
    parameterization in the full S4D paper -- this trades some
    expressiveness (a real diagonal system can't represent oscillatory
    modes as compactly as a complex one can) for a much simpler
    implementation with no complex-number pytree handling. Non-selective:
    contrast `SelectiveSSM` below, whose B/C/dt are computed from the
    input at every timestep instead of fixed per layer.

    Input/output: `(batch, seq_len, channels)` -> `(batch, seq_len, channels)`.

    Runs via `jax.lax.scan`, so it's O(seq_len) sequential steps -- a
    parallel/associative-scan implementation (as in the S4/S5 papers, for
    logarithmic rather than linear depth) is a possible future
    optimization, not done here.
    """

    channels: int
    state_dim: int = 16
    dt_min: float = 0.001
    dt_max: float = 0.1

    def setup(self):
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
        # x: (batch, seq_len, channels)
        A = -jnp.exp(self.log_A)               # (channels, state_dim), always < 0
        dt = jnp.exp(self.log_dt)               # (channels,)
        dA = jnp.exp(A * dt[:, None])           # (channels, state_dim)
        dB = (dA - 1.0) / A * self.B            # exact ZOH discretization of B

        batch = x.shape[0]
        h0 = jnp.zeros((batch, self.channels, self.state_dim), dtype=x.dtype)
        xt = jnp.transpose(x, (1, 0, 2))        # (seq_len, batch, channels)

        def step(h, u_t):
            h_new = dA[None] * h + dB[None] * u_t[:, :, None]
            y_t = jnp.sum(self.C[None] * h_new, axis=-1) + self.D[None] * u_t
            return h_new, y_t

        _, ys = jax.lax.scan(step, h0, xt)
        return jnp.transpose(ys, (1, 0, 2))     # (batch, seq_len, channels)


class SelectiveSSM(Module):
    """Selective SSM -- the S6 recurrence from Mamba (Gu & Dao, 2023):
    https://arxiv.org/abs/2312.00752

    Same diagonal per-channel linear state as `SSM` above, but B, C, and
    the timestep dt are no longer fixed per layer -- they're computed from
    the input at each timestep via learned projections, so the
    recurrence's effective dynamics can depend on content (the "selection
    mechanism" the name refers to, and the reason Mamba outperforms plain
    linear SSMs on tasks needing content-based reasoning).

    This implements the core selective-scan recurrence only, not the full
    Mamba block -- the reference architecture also wraps this in an input
    projection, a short causal depthwise `Conv` (see `conv.py`), and a
    SiLU gating branch. Composing those around this layer into a
    `MambaBlock`-style combinator is a natural next step, not included
    here.

    Discretization matches the reference Mamba implementation's choice:
    exact zero-order-hold for A (`dA = exp(A * dt)`, same as `SSM` above),
    but a simplified Euler step for B (`dB = dt * B`) rather than the exact
    ZOH form `(dA - 1) / A * B` -- this is a documented simplification in
    the original paper/code (the two coincide as dt -> 0), not an
    inconsistency with `SSM`'s discretization above.

    Input/output: `(batch, seq_len, d_inner)` -> `(batch, seq_len, d_inner)`.
    """

    d_inner: int
    state_dim: int = 16
    dt_rank: int = None

    def setup(self):
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
        # x: (batch, seq_len, d_inner)
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
            u, dt, B, C = inputs
            dA = jnp.exp(A[None] * dt[:, :, None])       # (batch, d_inner, state_dim)
            dB = dt[:, :, None] * B[:, None, :]            # (batch, d_inner, state_dim)
            h_new = dA * h + dB * u[:, :, None]
            y = jnp.sum(C[:, None, :] * h_new, axis=-1) + self.D[None] * u
            return h_new, y

        _, ys = jax.lax.scan(step, h0, (xt, dt_t, B_t, C_t))
        return jnp.transpose(ys, (1, 0, 2))  # (batch, seq_len, d_inner)


__all__ = ["SSM", "SelectiveSSM"]
