"""
Pure block-tiling helpers for tiled attention.

TODO: extract `_num_blocks` and `_block_bounds` from `xenafl_attention.py`
into this module. These are generic tiling utilities (block-index ->
start offset / position / validity mask) with no attention-specific or
masking-specific logic, so they belong here rather than inline in the
forward/backward passes.

`xenafl_attention.py`'s forward/backward should import these rather
than defining them inline once this extraction is done.
"""

from __future__ import annotations

__all__: list[str] = []
