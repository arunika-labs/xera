"""
Flash attention implementations.

Grouped separately from the rest of `xera.loom` because these are
kernel/dispatch-level attention implementations (backend selection +
kernels), not `Module` layers like the rest of the package. The public
functional entry point lives at `xera.functional.attention`
(re-exported as `xera.functional.auto_flash_attention`), sitting
alongside `xera.functional`'s other functions the way
`jax.nn.dot_product_attention` sits alongside the rest of `jax.nn` --
this package is the implementation those re-exports point to, not
itself the primary import path for users.

Layout:

    - `auto_flash_attention.py` -- the dispatcher. Picks a backend for
      the current device: Splash on TPU, cuDNN fused attention on GPU
      (sm_80+ by default), `xenafl_attention` as the portable fallback
      everywhere else (or whenever a vendor backend can't serve the
      request). "By default" because this only governs the routing
      choice when `backend` is left as `None` (auto-dispatch) --
      `xenafl_attention` is naive-but-portable by design and works
      correctly (just not at vendor-kernel speed) on any device/GPU
      arch when explicitly requested.
    - `xenafl_attention.py` -- the pure-jnp tiled flash attention kernel
      (block tiling + online softmax + custom_vjp) used as that
      fallback, and usable directly on its own.
    - `compat.py` -- device capability detection (NVIDIA compute
      capability, platform), shared by the dispatcher and by tests.
      Not yet implemented.
    - `core/` -- the individual pure-jnp pieces `xenafl_attention.py` is
      built from (tiling, masking, online softmax), factored out so
      they're independently usable/testable rather than only reachable
      as inline logic in `xenafl_attention.py`. Not yet extracted.

Three flash-attention backends exist in total: `xenafl` (portable,
pure-jnp, everywhere), `cudnn` (GPU, sm_80+), and `splash` (TPU).
"""

from __future__ import annotations

from .auto_flash_attention import auto_flash_attention
from .xenafl_attention import xenafl_attention

__all__ = [
    "auto_flash_attention",
    "xenafl_attention",
]
