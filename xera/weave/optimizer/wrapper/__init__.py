"""Wrapper factories -- each wraps a single Optimizer, generic across
whatever optimizer that is.

    clip.py         Clip
    schedule.py     Schedule
    accumulate.py   Accumulate

To add a new wrapper: drop a new file here following the factory pattern
(`__call__(inner: Optimizer) -> Optimizer`), then re-export it below.
"""

from .clip import Clip
from .schedule import Schedule
from .accumulate import Accumulate

__all__ = ["Clip", "Schedule", "Accumulate"]
