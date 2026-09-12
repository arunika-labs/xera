"""
Functional attention entry points.

Unlike `activations.py` in this same package, this module is not a thin
alias over `jax.nn` — it re-exports `flash_sdpa` from
`xera._kernel.flash_attention`, which is an original implementation.

The intent is for `flash_sdpa` to sit here at the same conceptual
position `jax.nn.dot_product_attention` occupies in `jax.nn`: a single
functional entry point for attention, reachable as
`xera.functional.flash_sdpa`, rather than living only under the
`flash_attention` implementation package.

The implementation itself -- backend dispatch and all three backend
kernels (cuDNN, splash, and the portable pure-jnp "jax" kernel) -- stays
private inside `xera._kernel.flash_attention`. This module is
deliberately just the public seam, same as `activations.py` is for the
`jax.nn` activation aliases.

Example:
    >>> import xera.functional as F
    >>> out = F.flash_sdpa(q, k, v, causal=True)
    >>> out = F.flash_sdpa(q, k, v, causal=True, backend="jax")
    >>> out = F.flash_sdpa(q, k, v, causal=True, verbose=True)  # print fallback reason, if any
"""

from __future__ import annotations

from .._kernel.flash_attention import flash_sdpa

__all__ = [
    "flash_sdpa",
]
