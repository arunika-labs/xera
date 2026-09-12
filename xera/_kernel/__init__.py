"""
Internal kernel/dispatch-level implementations.

This package (note the underscore, following the same convention as
`xera._rng`) is not part of the public API and its internal layout is
not guaranteed stable. It holds implementation details -- backend
dispatch, tiled kernels, masking/tiling internals -- that public seams
elsewhere (`xera.loom`, `xera.functional`) re-export from.

Currently contains:

    - `flash_attention/` -- flash attention backend dispatch
      (`flash_sdpa`), with the cuDNN/splash/portable-jax kernels behind
      it all private. See `xera._kernel.flash_attention` for details.
      Its public seam is `xera.functional.flash_sdpa`.
    - `shard.py` -- the device-sharding decorator (`shard`). Unlike
      `flash_attention`, its public seam is re-exported directly at
      the top level as `xera.shard`, since sharding cuts across
      `loom`, `functional`, and `optimizer` rather than belonging to
      one of them.
"""

from __future__ import annotations

from .shard import shard

__all__ = [
    "shard",
]
