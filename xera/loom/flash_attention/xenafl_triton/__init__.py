"""
XeraNaiveFlash, Triton edition -- the same naive-but-portable philosophy
as `xenafl_attention` (block tiling + online softmax, O(seq_len)
memory, not chasing peak vendor-kernel speed), implemented as an
`@triton.jit` kernel via `jax-triton` instead of pure `jnp`.

Primary motivation is filling the gap cuDNN leaves on pre-Ampere NVIDIA
GPUs (sm_70 Volta, sm_75 Turing), where no fused vendor kernel exists.
But like `xenafl_attention`, it is not restricted to that gap: Triton
compiles this kernel for whatever GPU it runs on, so it works (just
slower than cuDNN) on sm_80+ too, and remains composable there for
anyone who wants it explicitly -- see `dispatch.py`.

`kernel.py` holds the raw `@triton.jit` kernels; `dispatch.py` wires
them into the JAX-callable, differentiable `flash_attention_triton`.
Neither has been run against a real GPU yet -- see both modules'
docstrings for the untested-code caveat and remaining TODOs.

`jax-triton` is an optional dependency, only required if this backend
is actually reached; nothing at the top of `xera.loom` imports this
package eagerly.
"""

from __future__ import annotations

from .dispatch import flash_attention_triton

__all__ = ["flash_attention_triton"]
