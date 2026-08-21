"""
Device capability detection for backend dispatch.

Not yet implemented. `auto_flash_attention.py` currently decides
cuDNN-vs-fallback purely from dtype (see `_cudnn_compatibility_issue`)
and `platform == "gpu"` -- it does not yet check NVIDIA compute
capability at all, so a pre-Ampere GPU (sm_70/sm_75) with a supported
dtype will currently be routed to cuDNN (or fail there) rather than to
the Triton backend meant for it.

TODO:
    - `compute_capability(device) -> tuple[int, int] | None` -- the
      device's (major, minor) SM version for an NVIDIA GPU, or None for
      non-NVIDIA / non-GPU devices. (jax doesn't expose this directly;
      this will likely need `device.device_kind` string parsing or a
      `jaxlib`/CUDA runtime call -- needs investigation.)
    - `supports_cudnn_fused_attention(device) -> bool` -- True iff
      sm_80+. `auto_flash_attention`'s GPU branch should consult this
      *in addition to* the existing dtype check, and route to
      `xenafl_triton` (once implemented) rather than `xenafl_attention`
      when the device is sm_70/sm_75 specifically.

This module exists so device-capability logic lives in exactly one
place, shared by `auto_flash_attention.py` and by tests, rather than
duplicated/inlined at each call site.
"""

from __future__ import annotations

__all__: list[str] = []
