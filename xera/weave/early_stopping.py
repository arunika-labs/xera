"""
EarlyStopping: a concrete `Hook` that aborts training when a monitored
metric stops improving.

See `xera.weave.hook` for what a `Hook` is, and `xera.weave.callback` for
how `Callback.stop`/`XeraHook` actually abort training. This
module is the first concrete `Hook` built on both: it tracks a running
best for `monitor` as ordinary host-side state (`_best`, `_wait`, plain
Python floats/ints living on `self`, invisible to JAX tracing), and once
that metric hasn't improved by at least `min_delta` for `patience`
consecutive steps, it logs why (via `Metrics.log`, so the stop reason
ends up wherever your metrics already go) and calls `Callback.stop`.
"""

from __future__ import annotations
from .hook import Hook
from .metrics import Metrics
from .callback import Callback


class EarlyStopping(Hook):
    """
    Stop training when `monitor` stops improving for `patience` steps.

    Example:
        >>> from xera.weave import Callback, EarlyStopping
        >>>
        >>> hooks = [EarlyStopping(monitor="val_loss", patience=5)]
        >>>
        >>> def step(carry, i):
        ...     model, opt_state = carry
        ...     ...
        ...     Callback.run_hooks(i, hooks, loss=loss, val_loss=val_loss)
        ...     return (model, opt_state), loss
        >>>
        >>> try:
        ...     final, ys = loop.run(step, init_carry)
        ... except jax.errors.JaxRuntimeError as e:
        ...     if Callback.is_training_stopped(e):
        ...         print("stopped early:", e)
        ...     else:
        ...         raise

    Attributes:
        monitor: Name of the metric to watch, matched against the
            `logs` dict passed to `on_step_end` (e.g. `"loss"`,
            `"val_loss"`).
        patience: Number of consecutive non-improving steps to tolerate
            before stopping.
        mode: `"min"` if lower `monitor` values are better (the default
            -- e.g. a loss), `"max"` if higher is better (e.g. accuracy).
        min_delta: Minimum change to count as an improvement. A step
            whose `monitor` value improves by less than this still
            counts against `patience`.
    """

    monitor: str = "loss"
    patience: int = 5
    mode: str = "min"
    min_delta: float = 0.0

    def setup(self):
        """Validate config and initialize host-side tracking state."""
        assert self.mode in ("min", "max"), f"unknown mode: {self.mode!r}"
        assert self.patience >= 1, "EarlyStopping(patience=...) needs patience >= 1"
        object.__setattr__(self, "_best", None)
        object.__setattr__(self, "_wait", 0)

    def _improved(self, value):
        """Whether `value` counts as an improvement over `self._best`."""
        if self._best is None:
            return True
        if self.mode == "min":
            return value < self._best - self.min_delta
        return value > self._best + self.min_delta

    def on_step_end(self, step, logs):
        """
        Update the running best/wait counters and stop if patience is
        exhausted.

        Args:
            step: Current step (concrete Python int/scalar).
            logs: Dict of concrete metric values for this step; must
                contain `self.monitor` (silently does nothing if absent,
                e.g. on steps where a validation metric wasn't computed).
        """
        if self.monitor not in logs:
            return

        value = float(logs[self.monitor])

        if self._improved(value):
            object.__setattr__(self, "_best", value)
            object.__setattr__(self, "_wait", 0)
            return

        object.__setattr__(self, "_wait", self._wait + 1)
        if self._wait < self.patience:
            return

        # Log the stop reason before raising -- once Callback.stop fires,
        # nothing after it in this scan/jit call is guaranteed to run.
        Metrics.log(
            step=step,
            early_stopping_triggered=1.0,
            **{f"early_stopping_best_{self.monitor}": self._best},
        )
        Callback.stop(
            f"EarlyStopping: '{self.monitor}' did not improve for "
            f"{self.patience} steps (best={self._best}, last={value})"
        )


__all__ = ["EarlyStopping"]
