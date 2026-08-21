"""
Pure masking functions for tiled attention.

TODO: extract `_apply_masks` (and the padding/causal/local-window logic
it wraps) from `xenafl_attention.py` into standalone, composable
functions here:

    - padding_mask_fn(k_valid) -> mask
    - causal_mask_fn(q_positions, k_positions) -> mask
    - local_window_mask_fn(q_positions, k_positions, window_left, window_right) -> mask
    - segment_mask_fn(q_segment_ids, k_segment_ids) -> mask  (NEW: packing support,
      not yet implemented anywhere in this codebase)
    - combine_masks(*mask_fns) -> a single mask_fn

These should stay pure jnp, no dtype/platform assumptions -- same
ground rules `xenafl_attention.py`'s docstring already states for the
rest of that module. `xenafl_attention.py`'s forward/backward should
import and call these rather than defining masking inline once this
extraction is done.
"""

from __future__ import annotations

__all__: list[str] = []
