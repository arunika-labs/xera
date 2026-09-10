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
"""

from __future__ import annotations
