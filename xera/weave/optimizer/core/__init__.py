"""Core optimizers -- one gradient-transformation algorithm per file.

    sgd.py    SGDMomentum
    adam.py   AdamW (house other Adam variants here too, e.g. a future
              plain Adam or AdamW-8bit, rather than spinning up new files)
    lion.py   Lion
    muon.py   MuonCore, Muon

To add a new core optimizer: drop a new file here implementing the
`Optimizer` interface from `..base`, then re-export it below.
"""

from .sgd import SGDMomentum
from .adam import AdamW
from .lion import Lion
from .muon import MuonCore, Muon

__all__ = ["SGDMomentum", "AdamW", "Lion", "MuonCore", "Muon"]
