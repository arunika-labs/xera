"""xera.serialize -- single place to configure how things get persisted,
shared by both xera.loom (models) and xera.weave (optimizer/training
state), since a checkpoint is normally the pair of the two:

    serialize.save_model(model, "model.safetensors")
    serialize.save_state(opt_state, "opt_state.xera")
    ...
    model = serialize.load_model(template_model, "model.safetensors")
    opt_state = serialize.load_state("opt_state.xera")

Models go through `model.py` (safetensors -- portable, weights-only, safe
to share). State goes through `state.py` (xera's own pickle-based format
-- covers arbitrary pytree structure like optimizer moments and step
counters, meant for your own resume checkpoints, not for distribution).
"""

from .model import save_model, load_model
from .state import save_state, load_state

__all__ = ["save_model", "load_model", "save_state", "load_state"]
