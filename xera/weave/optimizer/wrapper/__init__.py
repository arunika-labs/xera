"""Wrapper factories -- each wraps a single Optimizer, generic across
whatever optimizer that is.

    clip.py           Clip
    schedule.py       Schedule
    accumulate.py     Accumulate
    weight_decay.py   WeightDecay
    ema.py            EMA
    freeze.py         Freeze
    lookahead.py      Lookahead
    cast.py           Cast

To add a new wrapper: drop a new file here following the factory pattern
(`__call__(inner: Optimizer) -> Optimizer`), then re-export it below.
"""

from .clip import Clip
from .schedule import Schedule
from .accumulate import Accumulate
from .weight_decay import WeightDecay
from .ema import EMA
from .freeze import Freeze
from .lookahead import Lookahead
from .cast import Cast

__all__ = [
    "Clip",
    "Schedule",
    "Accumulate",
    "WeightDecay",
    "EMA",
    "Freeze",
    "Lookahead",
    "Cast",
]
