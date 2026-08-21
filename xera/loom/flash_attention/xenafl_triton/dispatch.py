"""
Public entry point for the Triton backend (sm_70/sm_75 NVIDIA GPUs).

Not yet implemented. Once `kernel.py` exists, this module should expose
a `flash_attention_triton(q, k, v, *, causal=False, scale=None,
segment_ids=None, local_window_size=None)` function with the same
call shape as `xenafl_attention` and the cuDNN/splash wrappers in
`auto_flash_attention.py`, so `auto_flash_attention` can call it as a
drop-in dispatch target.

This module (rather than `kernel.py` directly) is what
`auto_flash_attention.py` should import from -- keeps the `jax-triton`
import lazy/contained here, so anyone not targeting sm_70/75 GPUs never
needs `jax-triton` installed at all.
"""

from __future__ import annotations

__all__: list[str] = []
