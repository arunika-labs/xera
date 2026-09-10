"""
Flash attention implementations.

Lives under `xera._kernel` (not `xera.loom`) because these are
kernel/dispatch-level attention implementations (backend selection +
kernels), not `Module` layers -- `xera.loom` is layers-only. The public
functional entry point lives at `xera.functional.attention`
(re-exported as `xera.functional.flash_sdpa`), sitting alongside
`xera.functional`'s other functions the way
`jax.nn.dot_product_attention` sits alongside the rest of `jax.nn` --
this package is the implementation those re-exports point to, not
itself the primary import path for users. All three backends (cuDNN,
splash, and the pure-jnp "jax" fallback) are private implementation
detail, reachable only through `flash_sdpa`'s `backend=` argument --
none of them, including the "jax" one, has a standalone public name.

Layout:

    - `flash_sdpa.py` -- the dispatcher, exposing `flash_sdpa`. Picks a
      backend for the current device: Splash on TPU, cuDNN fused
      attention on GPU (sm_80+ by default), the portable `"jax"` backend
      as the fallback everywhere else (or whenever a vendor backend
      can't serve the request). This is the behavior when
      `backend="auto"` (the default) -- the `"jax"` backend is
      naive-but-portable by design and works correctly (just not at
      vendor-kernel speed) on any device/GPU arch when explicitly
      requested via `backend="jax"`. Tile sizes for that backend
      (`block_q`/`block_k`) are configurable through `flash_sdpa`
      itself, so there's no need to reach the kernel module directly.
    - `jax_flash_attention.py` -- the pure-jnp tiled flash attention kernel
      (block tiling + online softmax + custom_vjp) used as that
      fallback. Private (`_jax_flash_attention`); not exported here or
      anywhere else, same as the cuDNN/splash implementations.
    - `compat.py` -- device capability detection (NVIDIA compute
      capability, platform), shared by the dispatcher and by tests.
      Not yet implemented.
    - `core/` -- the individual pure-jnp pieces `jax_flash_attention.py` is
      built from (tiling, masking, online softmax), factored out so
      they're independently usable/testable rather than only reachable
      as inline logic in `jax_flash_attention.py`. Not yet extracted.

Three flash-attention backends exist in total, selectable via
`flash_sdpa`'s `backend=` argument: `"jax"` (portable, pure-jnp,
everywhere), `"cudnn"` (GPU, sm_80+), and `"splash"` (TPU).
"""

from __future__ import annotations

from .flash_sdpa import flash_sdpa

__all__ = [
    "flash_sdpa",
]
