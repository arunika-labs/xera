"""
Flash attention implementations.

Grouped separately from the rest of `xera.loom` because these are
kernel/dispatch-level attention implementations (backend selection +
a pure-jnp tiled reference kernel), not `Module` layers like the rest
of the package.

    - `auto_flash_attention` -- picks a backend (Splash on TPU, cuDNN
      fused attention on GPU, `xenafl_attention` as the portable
      fallback everywhere else).
    - `xenafl_attention` -- the pure-jnp tiled flash attention kernel
      used as that fallback, and usable directly on its own.
"""

from __future__ import annotations

from .auto_flash_attention import auto_flash_attention
from .xenafl_attention import xenafl_attention

__all__ = [
    "auto_flash_attention",
    "xenafl_attention",
]
