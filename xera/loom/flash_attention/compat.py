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

Important: this capability check only ever affects `backend="auto"`'s
*default routing choice*. `xenafl_triton` (like `xenafl_attention`) is a
correct, portable kernel -- Triton compiles it for whatever GPU arch
it's run on, sm_70 through sm_90+. It is simply slower than cuDNN on
sm_80+, which is why "auto" prefers cuDNN there. A user who explicitly
passes `backend="xenafl_triton"` is asking for that specific kernel on
purpose (e.g. for its composability, or to test/benchmark it) and must
get it on any GPU, including sm_80+ -- this module's checks must never
gate or reject an explicitly-forced backend, only influence what
"auto" picks by default.

This module exists so device-capability logic lives in exactly one
place, shared by `auto_flash_attention.py` and by tests, rather than
duplicated/inlined at each call site.
"""

from __future__ import annotations

__all__: list[str] = []
