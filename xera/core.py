"""
Core module providing fundamental abstractions for the xera framework.

This module defines the base classes and utilities used throughout the framework:
- RNGPool: Manages random number generation for stochastic operations
- Buffer: A wrapper for values that should be treated as leaf nodes in JAX trees
- Module: Base class for all neural network components
- param: Helper function for parameter initialization
"""

from __future__ import annotations
import dataclasses
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


class RNGPool:
    """
    A pool for managing JAX random number generation keys.
    
    This class provides a convenient way to manage random keys for stochastic
    operations in neural networks. It maintains an internal key that gets
    split each time a new random key is requested, ensuring reproducible
    and independent random numbers.
    
    Attributes:
        _key: The internal JAX PRNG key.
    
    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> pool = RNGPool(key)
        >>> subkey1 = pool.next()
        >>> subkeys = pool.split(3)  # Get 3 independent keys
    """
    
    __slots__ = ("_key",)

    def __init__(self, key):
        """
        Initialize the RNG pool with a PRNG key.
        
        Args:
            key: A JAX PRNG key (typically from jax.random.PRNGKey).
        """
        self._key = key

    def next(self):
        """
        Get a single new random key from the pool.
        
        Returns:
            A new JAX PRNG key that can be used for random operations.
            The internal key is updated to ensure future calls return
            independent keys.
        """
        self._key, sub = jax.random.split(self._key)
        return sub

    def split(self, n):
        """
        Get multiple new random keys from the pool.
        
        Args:
            n: The number of random keys to generate.
        
        Returns:
            A list of n independent JAX PRNG keys. The internal key is
            updated to ensure future calls return independent keys.
        """
        self._key, *subs = jax.random.split(self._key, n + 1)
        return subs


class Buffer:
    """
    A wrapper for values that should be treated as leaf nodes in JAX trees.
    
    Buffer is used to wrap values that need to be part of the JAX pytree
    structure but should not be traversed further. This is particularly
    useful for running statistics in normalization layers and other
    stateful components.
    
    Attributes:
        value: The wrapped value.
    
    Example:
        >>> buffer = Buffer(jnp.zeros((10,)))
        >>> # The buffer will be treated as a leaf in JAX tree operations
    """
    

    __slots__ = ("value",)

    def __init__(self, value):
        """
        Initialize a buffer with a value.
        
        Args:
            value: The value to wrap in the buffer.
        """
        self.value = value

    def __repr__(self):
        """Return a string representation of the buffer."""
        return f"Buffer({self.value!r})"


jax.tree_util.register_pytree_with_keys(
    Buffer,
    lambda s: ([(jax.tree_util.GetAttrKey("value"), s.value)], None),
    lambda aux, children: Buffer(children[0]),
)


class Module:
    """
    Base class for all neural network modules in the xera framework.
    
    Module provides the foundation for building neural network components with:
    - Automatic JAX pytree registration for JIT compilation
    - Parameter initialization with random key management
    - Hierarchical module composition
    - Setup hooks for initialization logic
    
    Subclasses should define their parameters as class attributes and
    implement the __call__ method for forward computation.
    
    Example:
        >>> class MyLinear(Module):
        ...     in_features: int
        ...     out_features: int
        ...     
        ...     def setup(self):
        ...         self.weight = param(self.rng(), glorot_normal(), 
        ...                           (self.in_features, self.out_features))
        ...     
        ...     def __call__(self, x):
        ...         return x @ self.weight
        ...
        >>> layer = MyLinear(10, 20, key=jax.random.PRNGKey(0))
        >>> output = layer(jnp.ones((5, 10)))
    """
    

    def __init_subclass__(cls, **kwargs):
        """
        Automatically register subclasses as dataclasses and JAX pytrees.
        
        This method is called when a subclass is created and automatically:
        - Converts the class to a dataclass with specific configurations
        - Registers the class with JAX's pytree utilities for JIT compilation
        """
        super().__init_subclass__(**kwargs)
        cls = dataclasses.dataclass(cls, eq=False, repr=False, init=False)
        jax.tree_util.register_pytree_with_keys(
            cls,
            cls._tree_flatten_with_keys,
            cls._tree_unflatten,
            cls._tree_flatten,
        )

    def __new__(cls, *args, **kwargs):
        """Create a new instance of the module."""
        return super().__new__(cls)

    def __init__(self, *args, key=None, **kwargs):
        """
        Initialize the module with parameters and optional random key.
        
        Args:
            *args: Positional arguments corresponding to module fields.
            key: Optional JAX PRNG key for parameter initialization. If provided,
                enables the use of self.rng() during setup.
            **kwargs: Keyword arguments for module fields.
        
        Raises:
            RuntimeError: If self.rng() is called during setup but no key was provided.
        """
        field_names = [f.name for f in dataclasses.fields(self)]
        positional = dict(zip(field_names, args))
        for name, val in {**positional, **kwargs}.items():
            object.__setattr__(self, name, val)

        
        
        if key is not None:
            object.__setattr__(self, "_rng_pool", RNGPool(key))
        self.setup()
        if key is not None:
            object.__delattr__(self, "_rng_pool")  

    def setup(self):
        """
        Initialize module parameters and submodules.
        
        This method is called during __init__ and should be overridden by
        subclasses to initialize parameters and create submodules. Use
        self.rng() to get random keys for parameter initialization.
        
        Example:
            >>> def setup(self):
            ...     self.weight = param(self.rng(), normal(), (10, 20))
            ...     self.bias = param(self.rng(), zeros(), (20,))
        """
        pass

    def rng(self, n=None):
        """
        Get random keys from the module's RNG pool.
        
        Args:
            n: Optional number of random keys to generate. If None, returns
                a single key. If specified, returns n keys.
        
        Returns:
            A single JAX PRNG key if n is None, or a list of n keys.
        
        Raises:
            RuntimeError: If the module was created without a key parameter.
        """
        pool = getattr(self, "_rng_pool", None)
        if pool is None:
            raise RuntimeError(
                "self.rng() dipanggil tapi Module ini dibuat tanpa `key=`."
            )
        return pool.split(n) if n is not None else pool.next()

    def __call__(self, *args, **kwargs):
        """
        Forward pass of the module.
        
        This method must be implemented by subclasses to define the
        forward computation of the module.
        
        Args:
            *args: Input arguments to the module.
            **kwargs: Keyword arguments to the module.
        
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    def _tree_flatten(self):
        """
        Flatten the module for JAX pytree operations.
        
        This method separates the module's attributes into dynamic values
        (arrays, modules, buffers) that should be part of the pytree structure
        and static values (hyperparameters, configuration) that should remain
        constant during transformations.
        
        Returns:
            A tuple (dynamic_vals, aux_data) where:
                - dynamic_vals: Tuple of dynamic attribute values
                - aux_data: Auxiliary data for reconstruction (names, static values)
        """
        dynamic_names, dynamic_vals = [], []
        static_names, static_vals = [], []
        for name, val in self.__dict__.items():
            if isinstance(val, (jnp.ndarray, Module, Buffer)) or val is None:
                dynamic_names.append(name)
                dynamic_vals.append(val)
            elif isinstance(val, list) and val and all(isinstance(v, Module) for v in val):
                dynamic_names.append(name)
                dynamic_vals.append(val)
            else:
                static_names.append(name)
                static_vals.append(val)
        aux_data = (tuple(dynamic_names), tuple(static_names), tuple(static_vals))
        return tuple(dynamic_vals), aux_data

    def _tree_flatten_with_keys(self):
        """
        Flatten the module with attribute keys for JAX pytree operations.
        
        Similar to _tree_flatten, but includes attribute keys for better
        debugging and error messages in JAX transformations.
        
        Returns:
            A tuple (keyed_children, aux_data) where:
                - keyed_children: List of (key, value) pairs for dynamic attributes
                - aux_data: Auxiliary data for reconstruction
        """
        dynamic_vals, aux_data = self._tree_flatten()
        dynamic_names = aux_data[0]
        keyed = [
            (jax.tree_util.GetAttrKey(name), val)
            for name, val in zip(dynamic_names, dynamic_vals)
        ]
        return keyed, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a module from flattened representation.
        
        This is the inverse operation of _tree_flatten, used by JAX to
        reconstruct modules after transformations like JIT or grad.
        
        Args:
            aux_data: Auxiliary data containing attribute names and static values
            children: Dynamic attribute values from the flattened representation
        
        Returns:
            A reconstructed module instance.
        """
        dynamic_names, static_names, static_vals = aux_data
        obj = object.__new__(cls)
        for name, val in zip(dynamic_names, children):
            object.__setattr__(obj, name, val)
        for name, val in zip(static_names, static_vals):
            object.__setattr__(obj, name, val)
        return obj

    def __repr__(self):
        """
        Return a string representation of the module.
        
        Returns:
            A string showing the module class name and its field values.
        """
        field_names = [f.name for f in dataclasses.fields(self)]
        parts = ", ".join(f"{n}={getattr(self, n)!r}" for n in field_names)
        return f"{type(self).__name__}({parts})"


def param(key, init_fn, shape, dtype=jnp.float32):
    """
    Initialize a parameter using the given initialization function.
    
    This is a convenience function for creating parameters in modules.
    It applies the initialization function to generate parameter values
    with the specified shape and dtype.
    
    Args:
        key: A JAX PRNG key for random initialization.
        init_fn: An initialization function that takes (key, shape, dtype)
            and returns an initialized array. Common functions are defined
            in xera.initializers (e.g., normal(), xavier_normal(), etc.).
        shape: The shape of the parameter to create.
        dtype: The data type of the parameter (default: jnp.float32).
    
    Returns:
        An initialized parameter array with the specified shape and dtype.
    
    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> weight = param(key, initializers.xavier_normal(), (10, 20))
        >>> bias = param(key, initializers.zeros(), (20,))
    """
    return init_fn(key, shape, dtype)


# =============================================================================
# AutoFA (auto_flash_attention) -- differentiable core primitive.
#
# The naive flash-attention kernel used by `xera.loom.auto_flash_attention`
# is a Pallas kernel built from `fori_loop` + dynamic-start (`pl.ds`) reads.
# JAX's automatic reverse-mode differentiation cannot linearize through that
# combination (it raises "Linearization failed to produce known values for
# all output primals"), so this module can't just rely on `jax.grad` the way
# a plain jnp implementation could.
#
# Instead, this is a real `jax.custom_vjp`: `_autofa_forward` is the primal
# computation (a Pallas kernel, like before, but it additionally emits the
# per-query log-sum-exp `lse`), and `_autofa_backward` is a hand-written
# backward pass that recomputes attention probabilities block-by-block from
# `lse` -- the same algorithm the original FlashAttention paper uses for its
# backward pass, so the (seq_len, seq_len) score/probability matrix is never
# materialized in one shot, on either side.
#
# `xera.loom.auto_flash_attention` (the public API / backend dispatcher)
# calls into `auto_flash_attention` below for its naive-backend path; this
# is the only place the naive kernel's forward+backward math actually lives.
# =============================================================================


# Lane width lse's trailing dimension is padded to, purely so its Pallas
# TPU block shape (..., block_q, _LSE_LANE_PAD) satisfies Mosaic's "last
# dim divisible by 128" lowering requirement -- see the comment at its
# out_specs entry in `_autofa_forward`. Only lane 0 carries real data.
_LSE_LANE_PAD = 128


def _autofa_kernel(
    q_ref, k_ref, v_ref, bias_ref, o_ref, l_ref, *,
    block_k: int,
    seq_len: int,
    causal: bool,
    scale: float,
    has_bias: bool,
    window_left: int | None,
    window_right: int | None,
):
    """
    Pallas kernel body for one (batch, head, q_block) grid cell.

    Identical online-softmax tiling to the plain (non-differentiable)
    forward kernel, except it also writes `l_ref`: the per-query
    log-sum-exp `m_i + log(l_i)`. That single extra value per query row is
    exactly what `_autofa_backward` needs to recompute this block's
    attention probabilities later without recomputing the running-max/
    running-sum recurrence -- `exp(scores - lse)` reproduces the softmax
    probabilities directly.
    """
    q_block_idx = pl.program_id(2)
    block_q = q_ref.shape[2]
    head_dim = q_ref.shape[3]

    q = q_ref[0, 0, :, :] * scale

    m_i = jnp.full((block_q,), -jnp.inf, dtype=jnp.float32)
    l_i = jnp.zeros((block_q,), dtype=jnp.float32)
    acc = jnp.zeros((block_q, head_dim), dtype=jnp.float32)

    padded_seq_len = k_ref.shape[2]
    num_k_blocks = padded_seq_len // block_k

    def body(k_idx, carry):
        m_i, l_i, acc = carry
        k_start = k_idx * block_k

        k_block = k_ref[0, 0, pl.ds(k_start, block_k), :]
        v_block = v_ref[0, 0, pl.ds(k_start, block_k), :]

        scores = jnp.dot(q, k_block.T, preferred_element_type=jnp.float32)

        if has_bias:
            bias_block = bias_ref[0, 0, :, pl.ds(k_start, block_k)]
            scores = scores + bias_block.astype(jnp.float32)

        k_pos = k_start + jax.lax.iota(jnp.int32, block_k)[None, :]
        q_pos = q_block_idx * block_q + jax.lax.iota(jnp.int32, block_q)[:, None]

        in_bounds = k_pos < seq_len
        scores = jnp.where(in_bounds, scores, -jnp.inf)

        if causal:
            scores = jnp.where(q_pos >= k_pos, scores, -jnp.inf)

        if window_left is not None or window_right is not None:
            rel = q_pos - k_pos
            if window_left is not None:
                scores = jnp.where(rel <= window_left, scores, -jnp.inf)
            if window_right is not None:
                scores = jnp.where(-rel <= window_right, scores, -jnp.inf)

        m_ij = jnp.max(scores, axis=-1)
        m_new = jnp.maximum(m_i, m_ij)

        m_new_is_neg_inf = jnp.isneginf(m_new)
        p = jnp.where(
            m_new_is_neg_inf[:, None], 0.0, jnp.exp(scores - m_new[:, None])
        )
        alpha = jnp.where(m_new_is_neg_inf, 0.0, jnp.exp(m_i - m_new))

        l_new = alpha * l_i + jnp.sum(p, axis=-1)
        acc_new = acc * alpha[:, None] + jnp.dot(
            p.astype(v_block.dtype), v_block, preferred_element_type=jnp.float32
        )

        return m_new, l_new, acc_new

    def skip_body(k_idx, carry):
        return carry

    def loop_body(k_idx, carry):
        if causal:
            k_start = k_idx * block_k
            q_block_start = q_block_idx * block_q
            needed = k_start <= (q_block_start + block_q - 1)
            return jax.lax.cond(needed, body, skip_body, k_idx, carry)
        return body(k_idx, carry)

    m_i, l_i, acc = jax.lax.fori_loop(0, num_k_blocks, loop_body, (m_i, l_i, acc))

    l_i_safe = jnp.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_i_safe[:, None]
    # log-sum-exp of an all-masked row (l_i == 0) is -inf by definition;
    # `_autofa_backward` guards on this explicitly to avoid NaNs.
    lse = jnp.where(l_i == 0.0, -jnp.inf, m_i + jnp.log(l_i_safe))

    o_ref[0, 0, :, :] = out.astype(o_ref.dtype)
    l_ref[0, 0, :, 0] = lse


def _autofa_pad_inputs(q, k, v, bias, *, has_bias, block_q, block_k):
    """
    Pads q/k/v up to a common multiple of block_q/block_k (so every
    dynamic-start read in the kernel is in-bounds) and broadcasts+pads
    `bias` to match, or returns a small dummy bias when `has_bias` is
    False. Shared between the forward kernel call and the backward pass so
    both tile the sequence identically.
    """
    batch, num_heads, seq_len, head_dim = q.shape
    padded_len_q = pl.cdiv(seq_len, block_q) * block_q
    padded_len_k = pl.cdiv(seq_len, block_k) * block_k
    padded_len = max(padded_len_q, padded_len_k)
    pad_amount = padded_len - seq_len

    if pad_amount > 0:
        pad_qkv = [(0, 0), (0, 0), (0, pad_amount), (0, 0)]
        q = jnp.pad(q, pad_qkv)
        k = jnp.pad(k, pad_qkv)
        v = jnp.pad(v, pad_qkv)

    if has_bias:
        bias = jnp.broadcast_to(bias, (batch, num_heads, seq_len, seq_len))
        if pad_amount > 0:
            bias = jnp.pad(
                bias, [(0, 0), (0, 0), (0, pad_amount), (0, pad_amount)]
            )
    else:
        bias = jnp.zeros((1, 1, block_q, padded_len), dtype=q.dtype)

    return padded_len, q, k, v, bias, pad_amount


def _autofa_forward(
    q, k, v, bias, *,
    has_bias: bool,
    causal: bool,
    scale: float,
    window_left: int | None,
    window_right: int | None,
    block_q: int,
    block_k: int,
    interpret: bool,
):
    """
    Primal (forward) computation. Returns `(out, lse)`, both already
    cropped back to the caller's original `seq_len` -- `lse` is not part
    of AutoFA's public output, it exists purely as the residual
    `_autofa_backward` needs.
    """
    batch, num_heads, seq_len, head_dim = q.shape
    padded_len, q_p, k_p, v_p, bias_p, pad_amount = _autofa_pad_inputs(
        q, k, v, bias, has_bias=has_bias, block_q=block_q, block_k=block_k,
    )

    grid = (batch, num_heads, padded_len // block_q)
    kernel = functools.partial(
        _autofa_kernel,
        block_k=block_k,
        seq_len=seq_len,
        causal=causal,
        scale=scale,
        has_bias=has_bias,
        window_left=window_left,
        window_right=window_right,
    )
    bias_block_spec = (
        pl.BlockSpec((1, 1, block_q, padded_len), lambda b, h, i: (b, h, i, 0))
        if has_bias
        else pl.BlockSpec((1, 1, block_q, padded_len), lambda b, h, i: (0, 0, 0, 0))
    )

    out, lse = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, i: (b, h, i, 0)),
            pl.BlockSpec((1, 1, padded_len, head_dim), lambda b, h, i: (b, h, 0, 0)),
            pl.BlockSpec((1, 1, padded_len, head_dim), lambda b, h, i: (b, h, 0, 0)),
            bias_block_spec,
        ],
        out_specs=[
            pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, i: (b, h, i, 0)),
            # lse is logically (batch, num_heads, seq_len) -- one scalar per
            # query row -- but Pallas TPU's Mosaic lowering requires the
            # *last two* dims of every block shape to be divisible by
            # (8, 128). A trailing 1-D block of shape (block_q,) fails that
            # (its last two dims come out as (1, block_q), and 1 % 8 != 0).
            # We instead carry lse as a (..., seq_len, 1) array so the
            # block's last two dims are (block_q, 1), and pad the trailing
            # dim up to 128 lanes so it also satisfies the 128-divisibility
            # requirement; only column 0 is ever read/written.
            pl.BlockSpec((1, 1, block_q, _LSE_LANE_PAD), lambda b, h, i: (b, h, i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((batch, num_heads, padded_len, head_dim), q.dtype),
            jax.ShapeDtypeStruct((batch, num_heads, padded_len, _LSE_LANE_PAD), jnp.float32),
        ],
        interpret=interpret,
    )(q_p, k_p, v_p, bias_p)

    lse = lse[..., 0]
    if pad_amount > 0:
        out = out[:, :, :seq_len, :]
        lse = lse[:, :, :seq_len]
    return out, lse


def _autofa_backward_single(
    q, k, v, out, d_out, lse, bias, *,
    seq_len: int,
    padded_len: int,
    block_q: int,
    block_k: int,
    causal: bool,
    scale: float,
    window_left: int | None,
    window_right: int | None,
    has_bias: bool,
):
    """
    Analytic backward for one (batch, head) slice (called under a double
    `vmap` over batch and heads). Recomputes attention probabilities
    block-by-block from `lse` rather than differentiating through the
    forward Pallas kernel -- the standard FlashAttention backward
    algorithm. Memory per step is O(block_q * block_k), not O(seq_len^2):
    the (padded_len, padded_len) score/probability matrix is never formed.

    `has_bias` is a Python-level (static) bool, so the `bias`-related
    branches below are only ever traced when a bias was actually
    supplied -- when `has_bias` is False, `bias`/`dbias` are simply `None`
    throughout (a valid, empty pytree leaf), not a dummy array.
    """
    head_dim = q.shape[-1]
    num_q_blocks = padded_len // block_q
    num_k_blocks = padded_len // block_k

    q_scaled = q * scale
    d_rowsum = jnp.sum(out * d_out, axis=-1)  # D_t = O_t . dO_t, per query row

    dk0 = jnp.zeros_like(k)
    dv0 = jnp.zeros_like(v)
    dbias0 = jnp.zeros_like(bias) if has_bias else None

    def q_block_body(carry, q_idx):
        dk, dv, dbias = carry
        q_start = q_idx * block_q
        q_blk = jax.lax.dynamic_slice(q_scaled, (q_start, 0), (block_q, head_dim))
        d_out_blk = jax.lax.dynamic_slice(d_out, (q_start, 0), (block_q, head_dim))
        lse_blk = jax.lax.dynamic_slice(lse, (q_start,), (block_q,))
        d_blk = jax.lax.dynamic_slice(d_rowsum, (q_start,), (block_q,))
        q_pos = q_start + jnp.arange(block_q)

        dq_acc0 = jnp.zeros((block_q, head_dim), dtype=jnp.float32)

        def k_block_body(k_idx, carry2):
            dq_acc, dk, dv, dbias = carry2
            k_start = k_idx * block_k
            k_blk = jax.lax.dynamic_slice(k, (k_start, 0), (block_k, head_dim))
            v_blk = jax.lax.dynamic_slice(v, (k_start, 0), (block_k, head_dim))

            scores = jnp.dot(q_blk, k_blk.T, preferred_element_type=jnp.float32)
            if has_bias:
                bias_blk = jax.lax.dynamic_slice(
                    bias, (q_start, k_start), (block_q, block_k)
                )
                scores = scores + bias_blk.astype(jnp.float32)

            k_pos = k_start + jnp.arange(block_k)
            in_bounds = k_pos[None, :] < seq_len
            scores = jnp.where(in_bounds, scores, -jnp.inf)
            if causal:
                scores = jnp.where(q_pos[:, None] >= k_pos[None, :], scores, -jnp.inf)
            if window_left is not None or window_right is not None:
                rel = q_pos[:, None] - k_pos[None, :]
                if window_left is not None:
                    scores = jnp.where(rel <= window_left, scores, -jnp.inf)
                if window_right is not None:
                    scores = jnp.where(-rel <= window_right, scores, -jnp.inf)

            # Recompute P directly from the saved log-sum-exp -- this is
            # exact (not an approximation): exp(scores - lse) reproduces
            # the same softmax the forward kernel computed. Rows with
            # lse == -inf saw no valid (in-window/in-bounds/causal) key at
            # all; guard them to 0 to avoid exp(-inf - (-inf)) = NaN.
            lse_is_neg_inf = jnp.isneginf(lse_blk)
            p = jnp.where(
                lse_is_neg_inf[:, None], 0.0, jnp.exp(scores - lse_blk[:, None])
            )

            dv_blk = jnp.dot(
                p.T.astype(d_out_blk.dtype), d_out_blk, preferred_element_type=jnp.float32
            )
            dp = jnp.dot(d_out_blk, v_blk.T, preferred_element_type=jnp.float32)
            ds = p * (dp - d_blk[:, None])

            # scores = scale * (Q . K^T) [+ bias]; q_blk already carries
            # the `scale` factor (computed as q_scaled above), so:
            #   d(q_blk)/dQ = scale  -> dQ = scale * (ds @ K)
            #   d(scores)/dK = q_blk (already scaled)  -> dK = ds^T @ q_blk
            dq_acc = dq_acc + scale * jnp.dot(ds, k_blk, preferred_element_type=jnp.float32)
            dk_blk = jnp.dot(ds.T, q_blk, preferred_element_type=jnp.float32)

            dk = jax.lax.dynamic_update_slice(
                dk,
                jax.lax.dynamic_slice(dk, (k_start, 0), (block_k, head_dim)) + dk_blk.astype(dk.dtype),
                (k_start, 0),
            )
            dv = jax.lax.dynamic_update_slice(
                dv,
                jax.lax.dynamic_slice(dv, (k_start, 0), (block_k, head_dim)) + dv_blk.astype(dv.dtype),
                (k_start, 0),
            )
            if has_bias:
                dbias = jax.lax.dynamic_update_slice(
                    dbias,
                    jax.lax.dynamic_slice(dbias, (q_start, k_start), (block_q, block_k))
                    + ds.astype(dbias.dtype),
                    (q_start, k_start),
                )
            return dq_acc, dk, dv, dbias

        def skip_body(k_idx, carry2):
            return carry2

        def loop_body(k_idx, carry2):
            if causal:
                k_start = k_idx * block_k
                needed = k_start <= (q_start + block_q - 1)
                return jax.lax.cond(needed, k_block_body, skip_body, k_idx, carry2)
            return k_block_body(k_idx, carry2)

        dq_acc, dk, dv, dbias = jax.lax.fori_loop(
            0, num_k_blocks, loop_body, (dq_acc0, dk, dv, dbias)
        )
        return (dk, dv, dbias), dq_acc.astype(q.dtype)

    (dk, dv, dbias), dq_blocks = jax.lax.scan(
        q_block_body, (dk0, dv0, dbias0), jnp.arange(num_q_blocks)
    )
    dq = dq_blocks.reshape(padded_len, head_dim)
    return dq, dk, dv, dbias


def _unbroadcast(grad, target_shape):
    """Sum-reduces `grad` back down to `target_shape`, undoing whatever
    broadcasting `jnp.broadcast_to` applied on the way in -- the standard
    pattern for a custom_vjp backward when the primal broadcasts one of
    its differentiable arguments."""
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))
    reduce_axes = tuple(
        i for i, (g, t) in enumerate(zip(grad.shape, target_shape)) if t == 1 and g != 1
    )
    if reduce_axes:
        grad = grad.sum(axis=reduce_axes, keepdims=True)
    return grad.reshape(target_shape)


def _autofa_backward(
    residuals, d_out, *,
    has_bias: bool,
    causal: bool,
    scale: float,
    window_left: int | None,
    window_right: int | None,
    block_q: int,
    block_k: int,
):
    q, k, v, bias, out, lse = residuals
    batch, num_heads, seq_len, head_dim = q.shape

    padded_len, q_p, k_p, v_p, bias_p, pad_amount = _autofa_pad_inputs(
        q, k, v, bias, has_bias=has_bias, block_q=block_q, block_k=block_k,
    )
    if pad_amount > 0:
        pad_qkv = [(0, 0), (0, 0), (0, pad_amount), (0, 0)]
        out_p = jnp.pad(out, pad_qkv)
        d_out_p = jnp.pad(d_out, pad_qkv)
        lse_p = jnp.pad(lse, [(0, 0), (0, 0), (0, pad_amount)], constant_values=-jnp.inf)
    else:
        out_p, d_out_p, lse_p = out, d_out, lse

    per_bh = functools.partial(
        _autofa_backward_single,
        seq_len=seq_len,
        padded_len=padded_len,
        block_q=block_q,
        block_k=block_k,
        causal=causal,
        scale=scale,
        window_left=window_left,
        window_right=window_right,
        has_bias=has_bias,
    )

    if has_bias:
        dq_p, dk_p, dv_p, dbias_full = jax.vmap(jax.vmap(per_bh))(
            q_p, k_p, v_p, out_p, d_out_p, lse_p, bias_p
        )
    else:
        dq_p, dk_p, dv_p, _ = jax.vmap(
            jax.vmap(functools.partial(per_bh, bias=None))
        )(q_p, k_p, v_p, out_p, d_out_p, lse_p)
        dbias_full = None

    dq = dq_p[:, :, :seq_len, :] if pad_amount > 0 else dq_p
    dk = dk_p[:, :, :seq_len, :] if pad_amount > 0 else dk_p
    dv = dv_p[:, :, :seq_len, :] if pad_amount > 0 else dv_p

    if has_bias:
        dbias_full = dbias_full[:, :, :seq_len, :seq_len]
        dbias = _unbroadcast(dbias_full, bias.shape)
    else:
        # `bias` here is the small dummy array `_autofa_pad_inputs`/the
        # public wrapper substitutes when there is no real bias; its
        # cotangent is unused downstream but must match its shape.
        dbias = jnp.zeros_like(bias)

    return dq, dk, dv, dbias


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9, 10, 11))
def auto_flash_attention(
    q, k, v, bias, has_bias, causal, scale, window_left, window_right,
    block_q, block_k, interpret,
):
    """
    Differentiable AutoFA naive-kernel primitive.

    This is `xera.core`'s home for the actual forward/backward math behind
    `xera.loom.auto_flash_attention`'s naive backend -- see the module
    docstring above. `q, k, v, bias` are the differentiable arguments;
    everything else is a static (nondiff) configuration value. Pass a
    small dummy `bias` array with `has_bias=False` when no bias is wanted
    (see `xera.loom.auto_flash_attention._flash_attention_naive`, which is
    the public entry point that constructs these arguments and should
    normally be used instead of calling this directly).

    Args:
        q, k, v: (batch, num_heads, seq_len, head_dim) arrays.
        bias: Additive attention bias, broadcastable to
            (batch, num_heads, seq_len, seq_len), or a dummy array when
            `has_bias` is False (never read in that case).
        has_bias: Whether `bias` is real and should be applied/differentiated.
        causal: Whether to apply a causal mask.
        scale: Softmax scale (not None -- resolve the "1/sqrt(head_dim)"
            default before calling this).
        window_left, window_right: Local attention window bounds, or None
            for unbounded on that side.
        block_q, block_k: Sequence-dimension tile sizes.
        interpret: Whether to force Pallas interpret mode.

    Returns:
        Output array of shape (batch, num_heads, seq_len, head_dim).
    """
    out, _lse = _autofa_forward(
        q, k, v, bias,
        has_bias=has_bias, causal=causal, scale=scale,
        window_left=window_left, window_right=window_right,
        block_q=block_q, block_k=block_k, interpret=interpret,
    )
    return out


def _autofa_apply_fwd(
    q, k, v, bias, has_bias, causal, scale, window_left, window_right,
    block_q, block_k, interpret,
):
    out, lse = _autofa_forward(
        q, k, v, bias,
        has_bias=has_bias, causal=causal, scale=scale,
        window_left=window_left, window_right=window_right,
        block_q=block_q, block_k=block_k, interpret=interpret,
    )
    residuals = (q, k, v, bias, out, lse)
    return out, residuals


def _autofa_apply_bwd(
    has_bias, causal, scale, window_left, window_right, block_q, block_k,
    interpret, residuals, d_out,
):
    dq, dk, dv, dbias = _autofa_backward(
        residuals, d_out,
        has_bias=has_bias, causal=causal, scale=scale,
        window_left=window_left, window_right=window_right,
        block_q=block_q, block_k=block_k,
    )
    return dq, dk, dv, dbias


auto_flash_attention.defvjp(_autofa_apply_fwd, _autofa_apply_bwd)
