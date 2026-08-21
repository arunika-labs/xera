"""
Functional attention entry points.

Unlike `activations.py` in this same package, this module is not a thin
alias over `jax.nn` — it re-exports `auto_flash_attention` from
`xera.loom.flash_attention`, which is an original implementation.

The intent is for `auto_flash_attention` to sit here at the same
conceptual position `jax.nn.dot_product_attention` occupies in `jax.nn`:
a single functional entry point for attention, reachable as
`xera.loom.functional.auto_flash_attention` (and re-exported at
`xera.loom.auto_flash_attention` the same way `xera.loom` re-exports
everything else in this package), rather than living only under the
`flash_attention` implementation package.

The implementation itself -- backend dispatch, the portable `xenafl`
kernel, masking/tiling internals -- stays in `xera.loom.flash_attention`.
This module is deliberately just the public seam, same as
`activations.py` is for the `jax.nn` activation aliases.

Example:
    >>> from xera.loom import auto_flash_attention
    >>> out = auto_flash_attention(q, k, v, causal=True)
"""

from __future__ import annotations

from ..flash_attention import auto_flash_attention

__all__ = [
    "auto_flash_attention",
]
