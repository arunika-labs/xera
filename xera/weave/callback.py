

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
"""

from __future__ import annotations
from jax.experimental import io_callback
from .metrics import Metrics, _default_emit


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
    def save_state(cls, step, state, path_fn):
        """
        Checkpoint a training state (optimizer state, a `Train`/`State`
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


__all__ = ["Callback"]
