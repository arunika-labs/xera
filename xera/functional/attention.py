"""
Functional attention entry points.

Unlike `activations.py` in this same package, this module is not a thin
alias over `jax.nn` — it re-exports `sdpa_flash` from
`xera._kernel.flash_attention`, which is an original implementation.

The intent is for `sdpa_flash` to sit here at the same conceptual
position `jax.nn.dot_product_attention` occupies in `jax.nn`: a single
functional entry point for attention, reachable as
`xera.functional.sdpa_flash`, rather than living only under the
`flash_attention` implementation package.

The implementation itself -- backend dispatch, the portable `jax_flash_attention`
kernel, masking/tiling internals -- stays in `xera._kernel.flash_attention`.
This module is deliberately just the public seam, same as
`activations.py` is for the `jax.nn` activation aliases.

Example:
    >>> from xera.functional import sdpa_flash
    >>> out = sdpa_flash(q, k, v, causal=True)
    >>> out = sdpa_flash(q, k, v, causal=True, backend="jax")
    >>> out = sdpa_flash(q, k, v, causal=True, verbose=True)  # print fallback reason, if any
"""

from __future__ import annotations

from .._kernel.flash_attention import sdpa_flash

__all__ = [
    "sdpa_flash",
]
