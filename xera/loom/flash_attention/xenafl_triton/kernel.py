"""
FA1-style Triton kernel for pre-Ampere NVIDIA GPUs (sm_70 Volta, sm_75
Turing), invoked via `jax-triton`.

Not yet implemented. This is the backend meant to sit between cuDNN
(sm_80+) and `xenafl_attention` (the pure-jnp portable fallback) in
`auto_flash_attention`'s dispatch order:

    TPU              -> splash_attention
    GPU, sm_80+       -> cuDNN fused attention
    GPU, sm_70/sm_75  -> THIS module
    everything else   -> xenafl_attention (pure jnp, always works)

TODO:
    - Forward kernel: block-tiled QK^T -> online softmax -> PV,
      `@triton.jit`, following the same tiling/masking/online-softmax
      structure as `xenafl_attention.py` (see that module's docstring
      for why the two-pass forward+recompute-backward strategy is used)
      but written in `triton.language` rather than `jnp`.
    - Backward kernel: recompute probabilities from the saved LSE
      (mirrors `xenafl_attention._backward_single`), accumulate dQ/dK/dV.
    - Mirror `xera.loom.flash_attention.core`'s mask/online-softmax
      structure as small `@triton.jit` helper functions here (not
      literally the same Python functions -- Triton is a different
      language -- but structured 1:1, so a change to `core/masking.py`
      has an obvious corresponding place to update here). See the
      composability discussion in the project README for why this
      mirrors rather than shares code with `core/`.
    - Support: causal, segment_ids (packing), local window, as a closed
      set of `constexpr`-selected options -- not arbitrary user-supplied
      mask/bias functions (see README's "Composability model" section
      for the reasoning).
"""

from __future__ import annotations

__all__: list[str] = []
