"""
Exception hierarchy for the xera framework.

`XeraError` and `XeraHook` look similar (both are exceptions raised by
xera itself, both can surface wrapped by JAX when raised from inside an
`io_callback`) but mean opposite things, and are deliberately **not**
related by inheritance:

- `XeraError`: something is wrong -- invalid config, a misuse of the
  API, a state that should not be possible. An ordinary error.
- `XeraHook`: nothing is wrong. A `Hook`'s condition matched exactly as
  designed (early stopping's patience ran out, a NaN guard caught a
  NaN) and training is being deliberately halted as a result. It is
  xera's own control-flow signal, not a failure.

Keeping them siblings under `Exception` (rather than making `XeraHook` a
subclass of `XeraError`) means `except XeraError` can never accidentally
swallow a deliberate hook-triggered stop, and `except XeraHook` can
never accidentally swallow a real error -- the two are meant to be
handled differently, so catching one should never catch the other.

Both carry a `reason`/message that stays visible even after JAX
re-wraps the exception raised inside an `io_callback` (typically as
`jax.errors.JaxRuntimeError`) -- JAX's own wrapping already prefixes the
message with the exception's qualified class name (e.g.
`"xera.errors.XeraHook: <reason>"`), so `XeraHook`/`XeraError` don't
need to add their own prefix on top of that. See
`xera.weave.callback.Callback.is_training_stopped` for how the class
name surviving in `str(exc)` is used to tell a deliberate `XeraHook`
stop apart from any other error once the exception reaches the other
side of a `jax.lax.scan`/`jit` call.
"""

from __future__ import annotations


class XeraError(Exception):
    """
    Base class for ordinary errors raised by the xera framework itself.

    Use this (or a subclass) for genuine problems: invalid
    configuration, misuse of an API, an unreachable/inconsistent state.
    Not related to `XeraHook`, which signals a deliberate, by-design
    stop rather than a failure -- see the module docstring.
    """
    pass


class XeraHook(Exception):
    """
    Raised by a `Hook` (via `Callback.stop`) to deliberately abort
    training.

    Not a subclass of `XeraError`: an `XeraHook` firing means a stop
    condition matched exactly as designed (e.g. `EarlyStopping`'s
    patience ran out), not that something went wrong. See the module
    docstring for why the two are kept as separate hierarchies, and
    `xera.weave.hook`/`xera.weave.callback` for how `XeraHook` is
    raised and propagates out of a training loop.

    Attributes:
        reason: A human-readable explanation of why training stopped.
    """

    def __init__(self, reason="XeraHook"):
        self.reason = reason
        super().__init__(reason)


__all__ = ["XeraError", "XeraHook"]
