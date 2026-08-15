"""xera.weave.optimizer -- optimizers as composable building blocks.

    xera/weave/optimizer/
        base.py           Optimizer (base class), apply_updates
        partition.py      Partition (combinator over several optimizers --
                           doesn't fit core/ (not a single gradient algorithm)
                           or wrapper/ (doesn't wrap just one), stays top-level)
        core/             one core optimizer per file
            sgd.py            SGDMomentum
            adam.py           AdamW (and future Adam variants)
            lion.py           Lion
            muon.py           MuonCore, Muon (core + packed sugar)
        wrapper/          one wrapper per file
            clip.py           Clip
            schedule.py       Schedule
            accumulate.py     Accumulate

Wrappers (Clip, Schedule, Accumulate) are factories: construct with config,
then call on an optimizer to wrap it --

    opt = O.Clip(1.0)(O.Schedule(cosine_fn)(O.Muon(lr=0.02)))

Partition is a combinator over several optimizers, not a single-optimizer
wrapper, so it's constructed directly with its rules and used as-is.

To add a new core optimizer: drop a file in core/. To add a new wrapper:
drop a file in wrapper/. Either way, implement it against the `Optimizer`
interface in base.py and re-export it from the relevant __init__.py plus
here.
"""

from .base import Optimizer, apply_updates
from .partition import Partition
from .core import SGDMomentum, AdamW, Lion, MuonCore, Muon
from .wrapper import Clip, Schedule, Accumulate

__all__ = [
    "Optimizer",
    "apply_updates",
    "SGDMomentum",
    "AdamW",
    "Lion",
    "MuonCore",
    "Muon",
    "Clip",
    "Schedule",
    "Accumulate",
    "Partition",
]
