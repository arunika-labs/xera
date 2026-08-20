"""
Hook module: stateful stop-conditions for JAX-based training loops.

Two other pieces already cover the rest of a training loop's side
effects:

- `Callback` (see `xera.weave.callback`) runs *deterministic, ordered
  host effects* -- checkpointing, writing a log line -- as a direct
  function of the current step/values. It needs no memory between
  calls.
- Validation is not a callback or a hook at all: it runs inline, as
  part of the training step's own body (compute `val_loss`, fold it
  into `logs`), the same as computing the training loss.

`Hook` exists for what's left over: a decision that depends on
*history*, not just the current step -- "has this metric failed to
improve for the last N steps", "did a NaN just show up", "has the time
budget run out" -- and whose only real action is to abort training.
That's why `Hook` has exactly one lifecycle point, `on_step_end`, and
why the built-in `EarlyStopping` is the model for how a `Hook` should
look: track a small bit of host-side state (`best`, `wait`, ...),
compare it against the new step's `logs`, and if the condition fires,
call `Callback.stop` (usually after writing a `Metrics.log` entry
explaining why).

Because a `Hook` is a `Struct`, its state (counters, running bests) is
declared like any other `Struct` field, but lives as ordinary host-side
Python state -- mutated via `object.__setattr__` inside the
`io_callback` that `Callback.run_hooks` fires -- rather than as
something JAX traces or differentiates. If a check doesn't need memory
across steps, it almost certainly belongs in `Callback` (or inline in
the step, for validation) instead of as a `Hook`.

Example (a NaN guard, in the same spirit as `EarlyStopping`):
    >>> import jax.numpy as jnp
    >>> from xera.weave import Hook, Callback, Metrics
    >>>
    >>> class NaNGuard(Hook):
    ...     monitor: str = "loss"
    ...
    ...     def on_step_end(self, step, logs):
    ...         value = logs.get(self.monitor)
    ...         if value is not None and jnp.isnan(value):
    ...             Metrics.log(step=step, nan_guard_triggered=1.0)
    ...             Callback.stop(f"NaNGuard: '{self.monitor}' is NaN at step {step}")
    >>>
    >>> hooks = [NaNGuard(monitor="loss")]
    >>>
    >>> def train_step(carry, i):
    ...     model, opt_state = carry
    ...     ...
    ...     Callback.run_hooks(i, hooks, loss=loss)
    ...     return (model, opt_state), loss
"""

from __future__ import annotations
from .struct import Struct


class Hook(Struct):
    """
    Base class for stateful, history-dependent training stop-conditions.

    Not a general lifecycle system -- see the module docstring for why
    logging/checkpointing belong in `Callback` and validation belongs
    inline in the training step. `Hook` covers the narrower case where a
    check needs memory across steps to decide whether to abort training
    (early stopping, NaN guards, time/step budgets, and similar).

    `Callback.run_hooks` calls `on_step_end` on a list of hooks, in
    order, from within an ordered `io_callback` -- so a subclass's
    `on_step_end` runs as plain host-side Python: read `self`, compare
    against `logs`, optionally mutate `self` via `object.__setattr__`,
    and call `Callback.stop(reason)` if the condition is met.

    Example:
        >>> class StepBudget(Hook):
        ...     max_steps: int = 10_000
        ...
        ...     def on_step_end(self, step, logs):
        ...         if step >= self.max_steps:
        ...             Callback.stop(f"StepBudget: reached {self.max_steps} steps")
    """

    def on_step_end(self, step, logs):
        """
        Called once per training step, after the step's metrics are known.

        Args:
            step: The current step, as a concrete (non-traced) Python
                int -- this runs on the host via `io_callback`.
            logs: A dict of concrete metric values for this step (e.g.
                `{"loss": 0.42, "val_loss": 0.5}`), as passed to
                `Callback.run_hooks`.

        Override in subclasses to inspect `logs`/`self` and, if the
        stop condition is met, call `Callback.stop(reason)`. The
        default does nothing.
        """
        pass


__all__ = ["Hook"]
