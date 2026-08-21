"""
FA1-style Triton backend for pre-Ampere NVIDIA GPUs (sm_70, sm_75), via
`jax-triton`.

Not yet implemented -- see `kernel.py` and `dispatch.py` for scope.
`jax-triton` is an optional dependency, only required if this backend
is actually reached; nothing at the top of `xera.loom` should import
this package eagerly.
"""

from __future__ import annotations

__all__: list[str] = []
