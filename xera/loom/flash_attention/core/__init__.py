"""
Pure, composable algorithm pieces for tiled flash attention.

Everything in this package is plain `jnp` -- no dispatch, no vendor
backend, no `custom_vjp` -- just the individual pieces the algorithm is
made of, each usable and testable on its own:

    - `tiling.py`         -- block-index -> offset/position/validity helpers
    - `masking.py`        -- causal / local-window / padding / segment mask functions
    - `online_softmax.py` -- the running max/sum/accumulator update rule

`xenafl_attention.py` (one level up) is the consumer: it wires these
pieces together inside a `jax.lax.scan` and adds the `custom_vjp` needed
for O(seq_len) memory on the backward pass. Anyone wanting to build a
custom tiled-attention variant instead of using `xenafl_attention`
directly can import individual pieces from here.

Currently skeleton-only -- see each module's TODO for what still needs
to be extracted from `xenafl_attention.py`.
"""

from __future__ import annotations

__all__: list[str] = []
