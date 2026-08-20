

"""
I/O callback module for JAX-based training loops.

`Metrics` (see `xera.weave.metrics`) is built on `jax.debug.callback`: a
lightweight, best-effort hook meant for cheap, order-insensitive
side effects like printing a loss value. That's the right primitive for
logging, but it is explicitly *not* meant for operations with real,
must-happen side effects -- the docs describe it as a debugging tool, and
it gives no guarantee calls survive compiler optimizations or land in
program order relative to each other.

Checkpointing (writing a model or training state to disk) is a genuine
I/O effect: it has to actually happen, and if you fire off several
checkpoint writes across training steps, they have to land in the same
order those steps ran -- otherwise a later, smaller-numbered checkpoint
could overwrite a later one, or a write could simply vanish. For that,
`Callback` uses `jax.experimental.io_callback(..., ordered=True)` instead:
a native JAX primitive for impure host calls that preserves call order
across `jax.lax.scan`/`jax.jit`, without dropping out of the trace or
requiring any manual host synchronization.

The same gap shows up for metrics logging: `Metrics.log` is great for a
value that gets printed or shipped to a dashboard, where an occasional
dropped or reordered call doesn't matter. But sometimes you want a
*complete, ordered* record -- e.g. every registered metric appended as a
line to a training log file, so you can replay the run afterwards.
`Callback.log` gives you exactly that: it dispatches through the *same*
`Metrics._registry` (so you register emit functions once, with
`Metrics.register`, the usual way), but via `io_callback(ordered=True)`
instead of `jax.debug.callback` -- guaranteeing every entry is written,
in order. Use `Metrics.log` for cheap/best-effort output, `Callback.log`
when a metric's registered function has a real side effect (like a file
write) that has to fully happen and stay in step order.

Example:
    >>> from xera.serialize import save_model
    >>> from xera.weave import Callback
    >>>
    >>> def step(carry, i):
    ...     model, opt_state = carry
    ...     ...
    ...     jax.lax.cond(
    ...         i % 100 == 0,
    ...         lambda: Callback.save_model(i, model, lambda s: f"ckpt_{s}.safetensors"),
    ...         lambda: None,
    ...     )
    ...     return (model, opt_state), loss

Stopping training from a hook (see `xera.weave.hook.Hook`) uses the same
ordered-host-effect primitive, pushed one step further: a hook's
`on_step_end` can simply `raise XeraHook(reason)` (or call the
`Callback.stop(reason)` shortcut) from inside the `io_callback` that
`Callback.run_hooks` fires. JAX gives no first-class "stop this scan"
primitive, but a plain Python exception raised inside an `io_callback`
propagates out through it and out through the enclosing
`jax.lax.scan`/`jit` call exactly like it would out of any other Python
function call -- so `raise` is, in effect, JAX's stop mechanism here.
`XeraHook` (see `xera.errors`) exists specifically to mark this as a
*deliberate* stop rather than a bug: it is not a subclass of
`XeraError`, xera's ordinary-error base, precisely so the two can never
be accidentally caught by the same `except` clause. Two things are
worth knowing about the raise path:

- JAX re-wraps the exception (typically as a `jax.errors.JaxRuntimeError`)
  by the time it reaches your `except` block, so `except XeraHook`
  will *not* match. Catch the wrapped type and use
  `Callback.is_training_stopped(exc)` to check whether it was an
  `XeraHook` underneath, rather than parsing the traceback text
  yourself.
- The raise aborts the whole `scan`/`jit` call -- there's no "skip this
  step and continue" from here. Log whatever you need *before* raising
  (typically inside the same hook, via `Metrics.log`/`Callback.log`), not
  after, since after may never run.

Example:
    >>> from xera.weave import Callback, Hook
    >>>
    >>> class StopAtStep(Hook):
    ...     at: int = 100
    ...     def on_step_end(self, step, logs):
    ...         if step >= self.at:
    ...             Metrics.log(step=step, stopped=1.0)  # log first
    ...             Callback.stop(f"StopAtStep: reached step {step}")
    >>>
    >>> try:
    ...     final, ys = loop.run(step_fn, init_carry)
    ... except jax.errors.JaxRuntimeError as e:
    ...     if Callback.is_training_stopped(e):
    ...         print("training stopped early:", e)
    ...     else:
    ...         raise
"""

from __future__ import annotations
from jax.experimental import io_callback
from .metrics import Metrics, _default_emit
from ..errors import XeraHook


class Callback:
    """
    Registry of ordered I/O side-effect functions for JAX training loops.

    Where `Metrics.log` is for values you want printed/shipped to a
    dashboard and can afford to lose, `Callback` is for effects that must
    actually happen and must happen in order -- checkpointing chief among
    them. Every call goes through `jax.experimental.io_callback` with
    `ordered=True`.

    Attributes:
        _registry: Dictionary mapping callback names to host functions.

    Example:
        >>> @Callback.register("checkpoint")
        ... def _checkpoint(step, model):
        ...     save_model(model, f"ckpt_{int(step)}.safetensors")
        ...
        >>> Callback.call("checkpoint", step, model)
    """

    _registry = {}

    @classmethod
    def register(cls, name, fn=None):
        """
        Register a host function under a name.

        Can be used as a decorator or called directly, mirroring
        `Metrics.register`.

        Args:
            name: The callback name to register.
            fn: The host function. If None, returns a decorator.

        Returns:
            The function if fn is provided, otherwise a decorator.

        Example:
            >>> @Callback.register("checkpoint")
            ... def my_checkpoint(step, model):
            ...     save_model(model, f"ckpt_{int(step)}.safetensors")
        """
        def deco(f):
            cls._registry[name] = f
            return f
        if fn is not None:
            cls._registry[name] = fn
            return fn
        return deco

    @classmethod
    def unregister(cls, name):
        """
        Unregister a callback name.

        Args:
            name: The callback name to unregister.
        """
        cls._registry.pop(name, None)

    @classmethod
    def call(cls, name, step, *args, **kwargs):
        """
        Invoke a registered callback as an ordered `io_callback`.

        The registered function receives concrete (non-traced) values on
        the host -- `step` as a concrete scalar, and any pytree in `args`/
        `kwargs` (e.g. a model or training state) reconstructed with real
        array data -- so it's safe to do normal Python I/O inside it
        (file writes, `int()`/`float()` conversions for filenames, etc.).

        Args:
            name: The registered callback name.
            step: Current step, forwarded to the callback. Also what JAX
                uses to keep this call ordered relative to other ordered
                callbacks emitted from the same trace.
            *args: Extra pytree arguments forwarded to the callback (e.g.
                the model/state to checkpoint).
            **kwargs: Extra keyword pytree arguments forwarded likewise.

        Raises:
            KeyError: If no function is registered under `name`.

        Example:
            >>> Callback.call("checkpoint", step, model)
        """
        fn = cls._registry.get(name, None)
        if fn is None:
            raise KeyError(
                f"Callback: no function registered under {name!r}. "
                f"Register one first with Callback.register({name!r}, fn)."
            )
        io_callback(fn, None, step, *args, ordered=True, **kwargs)

    @classmethod
    def log(cls, step=None, **values):
        """
        Emit metric values through `Metrics`' registered emit functions,
        via an ordered `io_callback` instead of `jax.debug.callback`.

        This is `Metrics.log`'s durable counterpart: same registry
        (register emit functions with `Metrics.register` as usual, e.g. a
        function that appends a line to a log file), same call shape, but
        every call here is guaranteed to actually execute and to land in
        the same order the corresponding steps ran -- which matters once
        the registered function has a real side effect (a file write)
        instead of just a `print`.

        Args:
            step: Optional training step number.
            **values: Keyword arguments of metric names and values.

        Example:
            >>> @Metrics.register("loss")
            ... def _to_file(step, value):
            ...     with open("train.log", "a") as f:
            ...         f.write(f"{step},{value}\\n")
            ...
            >>> Callback.log(step=i, loss=loss)  # durable, ordered write
        """
        def _emit(step_, values_):
            for name, value in values_.items():
                fn = Metrics._registry.get(name, None)
                if fn is not None:
                    fn(step_, value)
                else:
                    _default_emit(name, step_, value)
        io_callback(_emit, None, step, values, ordered=True)

    @classmethod
    def save_model(cls, step, module, path_fn):
        """
        Checkpoint a model to safetensors, ordered so writes triggered
        across training steps land on disk in step order.

        This is a thin convenience wrapper around `xera.serialize.save_model`
        -- it does not go through the `_registry`, so it works without any
        prior `Callback.register` call.

        Args:
            step: Current training step (a traced int; concrete by the
                time the write actually runs on the host).
            module: The model (a JAX pytree, e.g. an `xera.core.Module`)
                to save.
            path_fn: A plain Python callable `int -> str` mapping a
                concrete step number to a file path (e.g.
                `lambda s: f"ckpt_{s}.safetensors"`). Runs on the host, so
                ordinary Python string formatting is fine here.

        Example:
            >>> Callback.save_model(i, model, lambda s: f"ckpt_{s}.safetensors")
        """
        from ..serialize import save_model as _save_model

        def _write(step_, module_):
            _save_model(module_, path_fn(int(step_)))

        io_callback(_write, None, step, module, ordered=True)

    @classmethod
    def stop(cls, reason="XeraHook"):
        """
        Raise `XeraHook` from the host to deliberately abort training.

        A thin, explicit `raise` -- provided as a named method mainly so
        call sites read as "stop training" rather than a bare `raise`,
        and so the exception type lives in one place. Call this from
        inside a `Hook`'s `on_step_end` (i.e. from code already running
        on the host inside an `io_callback`), typically *after* writing
        whatever log entry explains the stop -- once this raises,
        nothing after it in the same `scan`/`jit` call is guaranteed to
        run.

        Args:
            reason: Human-readable explanation, carried on the raised
                `XeraHook` and visible (wrapped, prefixed with
                `"XeraHook: "`) in the exception that reaches the
                caller of the training loop.

        Raises:
            XeraHook: Always.

        Example:
            >>> class EarlyStopping(Hook):
            ...     def on_step_end(self, step, logs):
            ...         if should_stop:
            ...             Metrics.log(step=step, early_stopped=1.0)
            ...             Callback.stop(f"EarlyStopping: no improvement")
        """
        raise XeraHook(reason)

    @staticmethod
    def is_training_stopped(exc):
        """
        Check whether a caught exception was an `XeraHook` raised from
        inside an `io_callback`.

        JAX re-wraps exceptions raised inside `io_callback` (typically as
        `jax.errors.JaxRuntimeError`), so the exception a caller catches
        around `loop.run(...)` is not an `XeraHook` instance -- a plain
        `except XeraHook` will never match. This checks the wrapped
        exception's message for `XeraHook`'s class name instead, which
        survives the wrapping (`XeraHook.__init__` always prefixes its
        message with `"XeraHook: "` for exactly this reason).

        Args:
            exc: The exception caught around the training loop call.

        Returns:
            True if `exc` (or its chained cause) originated from an
            `XeraHook` raised via `Callback.stop`/a `Hook` -- i.e. a
            deliberate stop, not an ordinary error.

        Example:
            >>> try:
            ...     final, ys = loop.run(step_fn, init_carry)
            ... except jax.errors.JaxRuntimeError as e:
            ...     if Callback.is_training_stopped(e):
            ...         print("stopped:", e)
            ...     else:
            ...         raise
        """
        return "XeraHook" in str(exc)

    @classmethod
    def run_hooks(cls, step, hooks, **logs):
        """
        Run `on_step_end` on a list of stop-condition `Hook`s, in order,
        via a single ordered `io_callback`.

        Only for `Hook`s (see `xera.weave.hook`) -- history-dependent
        stop conditions like `EarlyStopping`. Deterministic per-step
        effects (printing, logging to a file, checkpointing) belong in
        `Callback.log`/`Callback.call` instead, not here.

        All hooks for a given step run inside one host call, in list
        order, with concrete (non-traced) `step` and `logs` values -- so
        a hook's `on_step_end` can do normal Python: compare metrics
        against its own state, mutate that state
        (`object.__setattr__(self, ...)`), and call `Callback.stop` to
        abort training if its condition is met (see the module
        docstring for how that propagates).

        Args:
            step: Current step, forwarded to each hook's `on_step_end`.
            hooks: A list of `Hook` instances, called in order.
            **logs: Metric values for this step (e.g. `loss=loss`),
                forwarded as a single `logs` dict to each hook.

        Example:
            >>> hooks = [EarlyStopping(patience=3), NaNGuard()]
            >>> Callback.run_hooks(i, hooks, loss=loss, val_loss=val_loss)
        """
        def _run(step_, logs_):
            for hook in hooks:
                hook.on_step_end(step_, logs_)
        io_callback(_run, None, step, logs, ordered=True)

    @classmethod
    def save_state(cls, step, state, path_fn):
        """
        Checkpoint a training state (optimizer state, a `Struct`
        instance, or any other pytree) to safetensors, ordered so writes
        triggered across training steps land on disk in step order.

        This is a thin convenience wrapper around `xera.serialize.save_state`
        -- it does not go through the `_registry`, so it works without any
        prior `Callback.register` call.

        Args:
            step: Current training step (a traced int; concrete by the
                time the write actually runs on the host).
            state: The state (a JAX pytree) to save.
            path_fn: A plain Python callable `int -> str` mapping a
                concrete step number to a file path.

        Example:
            >>> Callback.save_state(i, opt_state, lambda s: f"opt_{s}.safetensors")
        """
        from ..serialize import save_state as _save_state

        def _write(step_, state_):
            _save_state(state_, path_fn(int(step_)))

        io_callback(_write, None, step, state, ordered=True)


__all__ = ["Callback", "XeraHook"]
