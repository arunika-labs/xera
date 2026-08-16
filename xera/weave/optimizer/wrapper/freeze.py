"""Freeze: leaves matching a predicate get a permanent zero update
(effectively frozen), everything else trains normally. Complementary to
Partition -- Partition routes different leaves to different optimizers,
Freeze routes some leaves to "no optimizer at all". Implemented as sugar
over Partition with a no-op optimizer for the frozen group, rather than a
separate masking implementation, for the same single-source-of-truth
reason `Muon(...)` is sugar over `Partition` + `MuonCore`.
"""

from __future__ import annotations
from typing import NamedTuple
import jax.numpy as jnp
from ..base import Optimizer, _tree_map
from ..partition import Partition


class _NoOpState(NamedTuple):
    step: jnp.ndarray


class _NoOp(Optimizer):
    """Always emits an all-zero update. Internal to Freeze -- not exported."""

    def init(self, params):
        return _NoOpState(step=jnp.zeros([], jnp.int32))

    def update(self, grads, state, params=None, step=None):
        zero = _tree_map(jnp.zeros_like, grads)
        return zero, _NoOpState(step=state.step + 1)


class Freeze:
    """Factory: `predicate(path, leaf) -> bool` selects leaves to freeze;
    everything else routes to the wrapped optimizer.

    Usage:
        opt = O.Freeze(lambda path, leaf: "backbone" in str(path))(O.AdamW(lr=1e-4))

    Equivalent by hand:
        opt = O.Partition([
            (lambda path, leaf: "backbone" in str(path), <no-op>),
            (lambda path, leaf: True, O.AdamW(lr=1e-4)),
        ])
    """

    def __init__(self, predicate):
        self.predicate = predicate

    def __call__(self, inner: Optimizer) -> Optimizer:
        return Partition([
            (self.predicate, _NoOp()),
            (lambda path, leaf: True, inner),
        ])


__all__ = ["Freeze"]
