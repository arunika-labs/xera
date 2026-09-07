"""
Flash attention implementations.

Lives under `xera._kernel` (not `xera.loom`) because these are
kernel/dispatch-level attention implementations (backend selection +
kernels), not `Module` layers -- `xera.loom` is layers-only. The public
functional entry point lives at `xera.functional.attention`
(re-exported as `xera.functional.sdpa_flash`), sitting alongside
`xera.functional`'s other functions the way
`jax.nn.dot_product_attention` sits alongside the rest of `jax.nn` --
this package is the implementation those re-exports point to, not
itself the primary import path for users. `xera.loom` also re-exports
`jax_flash_attention` directly (as `xera.loom.jax_flash_attention`) for
backward compatibility.

Layout:

    - `sdpa_flash.py` -- the dispatcher, exposing `sdpa_flash`. Picks a
      backend for the current device: Splash on TPU, cuDNN fused
      attention on GPU (sm_80+ by default), the portable `"jax"` backend
      (`jax_flash_attention`) as the fallback everywhere else (or
      whenever a vendor backend can't serve the request). This is the
      behavior when `backend="auto"` (the default) -- `jax_flash_attention`
      is naive-but-portable by design and works correctly (just not at
      vendor-kernel speed) on any device/GPU arch when explicitly
      requested via `backend="jax"`.
    - `jax_flash_attention.py` -- the pure-jnp tiled flash attention kernel
      (block tiling + online softmax + custom_vjp) used as that
      fallback, and usable directly on its own.
    - `compat.py` -- device capability detection (NVIDIA compute
      capability, platform), shared by the dispatcher and by tests.
      Not yet implemented.
    - `core/` -- the individual pure-jnp pieces `jax_flash_attention.py` is
      built from (tiling, masking, online softmax), factored out so
      they're independently usable/testable rather than only reachable
      as inline logic in `jax_flash_attention.py`. Not yet extracted.

Three flash-attention backends exist in total, selectable via
`sdpa_flash`'s `backend=` argument: `"jax"` (portable, pure-jnp,
everywhere), `"cudnn"` (GPU, sm_80+), and `"splash"` (TPU).
"""

from __future__ import annotations

from .sdpa_flash import sdpa_flash
from .jax_flash_attention import jax_flash_attention

__all__ = [
    "sdpa_flash",
    "jax_flash_attention",
]
