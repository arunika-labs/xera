"""Core optimizers -- one gradient-transformation algorithm per file
(variants of the same family share a file, e.g. Adam/AdamW in adam.py).

    sgd.py         SGDMomentum
    adam.py        Adam, AdamW
    lion.py        Lion
    muon.py        MuonCore, Muon
    rmsprop.py     RMSprop
    adagrad.py     Adagrad
    adan.py        Adan
    adafactor.py   Adafactor
    shampoo.py     Shampoo

To add a new core optimizer: drop a new file here implementing the
`Optimizer` interface from `..base`, then re-export it below. A new
variant of an existing family (another Adam-like optimizer, say) goes in
that family's existing file rather than a new one.
"""

from .sgd import SGDMomentum
from .adam import Adam, AdamW
from .lion import Lion
from .muon import MuonCore, Muon
from .rmsprop import RMSprop
from .adagrad import Adagrad
from .adan import Adan
from .adafactor import Adafactor
from .shampoo import Shampoo

__all__ = [
    "SGDMomentum",
    "Adam",
    "AdamW",
    "Lion",
    "MuonCore",
    "Muon",
    "RMSprop",
    "Adagrad",
    "Adan",
    "Adafactor",
    "Shampoo",
]
