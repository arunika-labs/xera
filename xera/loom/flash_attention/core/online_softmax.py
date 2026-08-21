"""
Pure online (running) softmax update rule.

TODO: extract the running max/sum/accumulator recurrence currently
inlined in `_forward_single`'s `k_block_step` (in `xenafl_attention.py`)
into a standalone function here, e.g.:

    online_softmax_update(m_prev, l_prev, acc_prev, scores_block, v_block)
        -> (m_new, l_new, acc_new)

This is the single most reusable piece of the algorithm -- it's the
same recurrence regardless of what masking/bias/tiling strategy sits
around it, so it should be callable on its own for anyone building a
custom tiled-attention variant, and independently testable against a
plain (non-tiled) softmax reference.

`xenafl_attention.py`'s forward pass should call this rather than
inlining the recurrence once this extraction is done.
"""

from __future__ import annotations

__all__: list[str] = []
