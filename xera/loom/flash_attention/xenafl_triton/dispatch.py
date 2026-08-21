"""
Public entry point for the Triton backend: `flash_attention_triton`.

Wires the kernels in `kernel.py` into a JAX-callable, differentiable
function via `jax_triton.triton_call` + `jax.custom_vjp` -- the same
external shape as `xenafl_attention`, so `auto_flash_attention` can call
either one interchangeably.

Despite being written with sm_70/sm_75 (pre-Ampere) GPUs as the primary
motivation, this backend is not restricted to them. Triton compiles the
same kernel for whatever GPU architecture it actually runs on, so this
function must work correctly on sm_80+ GPUs too when called directly
-- it will simply be slower than cuDNN there, not incompatible. See
`compat.py` and this project's README ("Composability model") for why
`auto_flash_attention`'s default routing still prefers cuDNN on sm_80+
without that meaning this backend is gated to sm_70/sm_75 only.

** UNTESTED, same caveat as `kernel.py`: there is no NVIDIA GPU in the
environment this was written in. This module has not been executed. **

Supported (mirrors xenafl_attention.py, minus `bias` -- see kernel.py's
module docstring for why bias is deferred):
    - causal masking
    - local window (window_left, window_right)
    - segment_ids (packing)

TODO before this is usable:
    - Implement `flash_attention_triton` below (currently NotImplementedError).
    - Choose default BLOCK_Q/BLOCK_K/BLOCK_D/num_warps/num_stages, or an
      autotuning strategy (`triton.autotune`) -- cannot be tuned without
      a real GPU, so defaults here should be conservative placeholders
      clearly marked as such, not treated as final.
    - Decide head_dim handling: `BLOCK_D` here is meant to equal
      `head_dim` directly (load the whole head dimension per block, as
      is standard for FA1-style kernels since head_dim is typically
      <=128) -- confirm this holds for whatever head_dims this project
      needs to support, and handle non-power-of-2 head_dim (padding to
      the next power of 2 for `tl.arange`, masking the excess) if so.
    - `jax.custom_vjp` wiring: forward calls `_xenafl_triton_fwd_kernel`
      via `triton_call`, saves `(q, k, v, out, lse, segment_ids)` as
      residuals; backward calls `_xenafl_triton_bwd_preprocess` then
      `_xenafl_triton_bwd_kernel` (after zero-initializing a float32 dQ
      buffer, per `kernel.py`'s docstring), and casts dQ back to `q`'s
      dtype before returning.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def flash_attention_triton(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool = False,
    scale: float | None = None,
    segment_ids: jax.Array | None = None,
    window_left: int | None = None,
    window_right: int | None = None,
    block_q: int = 128,
    block_k: int = 128,
) -> jax.Array:
    """
    XeraNaiveFlash attention, Triton edition: FA1-style fused kernel via
    `jax-triton`. Same algorithm and O(seq_len) memory guarantee as
    `xenafl_attention`, but a real fused GPU kernel rather than a
    sequence of separate XLA ops -- meant for NVIDIA GPUs below sm_80
    (Volta sm_70, Turing sm_75), where no vendor fused-attention kernel
    exists. Also runs correctly (just slower than cuDNN) on sm_80+ GPUs
    if called explicitly -- see this module's docstring.

    Not yet implemented -- see this module's TODO for the remaining work.

    Args:
        q, k, v: (batch, num_heads, seq_len, head_dim) arrays.
        causal: If True, apply a causal mask.
        scale: Softmax scale. If None, defaults to 1/sqrt(head_dim).
        segment_ids: Optional (batch, seq_len) int array. Query i and
            key j attend to each other only if
            segment_ids[..., i] == segment_ids[..., j] -- packing
            support, so multiple examples can share one padded sequence
            without cross-attending. None means no packing (equivalent
            to every position sharing one segment).
        window_left, window_right: Local attention window bounds, same
            semantics as `xenafl_attention`.
        block_q, block_k: Tile sizes. Defaults are placeholders (see
            TODO) -- not yet tuned against real hardware.

    Returns:
        Output array of shape (batch, num_heads, seq_len, head_dim),
        same dtype as `q`.
    """
    raise NotImplementedError(
        "flash_attention_triton is a skeleton -- the custom_vjp wiring "
        "to kernel.py's Triton kernels via jax_triton.triton_call is not "
        "yet implemented, and none of it has been run against a real "
        "GPU. See this module's docstring for the remaining TODO."
    )


__all__ = ["flash_attention_triton"]
