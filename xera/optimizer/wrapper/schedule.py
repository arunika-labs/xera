

from __future__ import annotations
from typing import Callable, Union
import jax.numpy as jnp
from ..base import Optimizer, _get_hyper, _set_hyper
from ..state import State


class Schedule(State):
    """
    Makes one or more of an inner optimizer's hyperparameters follow a
    `step -> value` schedule, by overwriting that hyperparameter's leaf
    in the inner optimizer's state before each `update()`.

    Every numeric hyperparameter on a core optimizer (`lr`, `b1`,
    `weight_decay`, ...) is already a leaf of the state it returns --
    carried forward unchanged, call after call, by default. Without
    wrapping in `Schedule`, that leaf just keeps its constructor value
    for the whole run. `Schedule` is what turns a schedule *on* for one
    (or several) of those leaves, generically, for any inner optimizer
    -- the inner optimizer itself never needs to know a schedule is
    involved, which is what keeps this composable with the rest of the
    wrapper chain (`Clip`, `WeightDecay`, `Partition`, ...).

    Because only the *value* of the targeted leaf changes each step
    (never the pytree structure, never anything on the `Optimizer`
    object itself), this never triggers an XLA recompile no matter how
    often the schedule's output changes.

    Args:
        fn: Either a single `step -> value` callable, applied to
            `target` (`"lr"` by default), or a dict `{name: step ->
            value}` to schedule several hyperparameters at once --
            including ones on a different layer of the wrapper chain,
            e.g. both the inner optimizer's `lr` and an outer
            `WeightDecay`'s `rate`.
        target: Which hyperparameter to schedule when `fn` is a single
            callable. Ignored when `fn` is a dict.

    Example:
        >>> opt = Schedule(lambda step: 1e-3 * 0.999 ** step)(Adam(lr=1e-3))
        >>> # schedule more than one hyperparameter at once:
        >>> opt = Schedule({
        ...     "lr": lambda step: 1e-3 * 0.999 ** step,
        ...     "weight_decay": lambda step: 1e-4 * (step < 1000),
        ... })(AdamW(lr=1e-3, weight_decay=1e-4))
    """

    fn: Union[Callable, dict] = None
    target: str = "lr"

    def setup(self):
        self._targets = dict(self.fn) if isinstance(self.fn, dict) else {self.target: self.fn}

    def __call__(self, inner: Optimizer) -> Optimizer:
        return _Scheduled(inner, self._targets)


class _Scheduled(Optimizer):
    inner: Optimizer = None
    targets: dict = None  # name -> `step -> value` callable

    def init(self, params):
        return self.inner.init(params)

    def update(self, grads, state, params=None, step=None):
        cur_step = _get_hyper(state, "step")
        use_step = cur_step if step is None else jnp.asarray(step)
        for name, fn in self.targets.items():
            state = _set_hyper(state, name, fn(use_step))
        return self.inner.update(grads, state, params, step)


__all__ = ["Schedule"]
