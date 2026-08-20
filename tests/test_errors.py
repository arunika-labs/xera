"""Tests for xera.errors: the XeraError / XeraHook exception hierarchy."""

import jax
import jax.numpy as jnp
import pytest
import xera
from xera.errors import XeraError, XeraHook
from xera.weave.hook import Hook
from xera.weave.callback import Callback
from xera.weave.loop import Loop


def test_errors_module_accessible_from_xera_namespace():
    assert xera.errors is not None
    assert xera.errors.XeraError is XeraError
    assert xera.errors.XeraHook is XeraHook


def test_xera_error_is_a_plain_exception():
    assert issubclass(XeraError, Exception)


def test_xera_hook_is_a_plain_exception():
    assert issubclass(XeraHook, Exception)


def test_xera_hook_is_not_a_subclass_of_xera_error():
    # Deliberate: a stop condition firing (XeraHook) is not the same
    # kind of thing as an ordinary error (XeraError). Keeping them as
    # siblings under Exception means `except XeraError` can never
    # accidentally swallow a deliberate hook-triggered stop, and vice
    # versa.
    assert not issubclass(XeraHook, XeraError)
    assert not issubclass(XeraError, XeraHook)


def test_xera_error_can_be_raised_and_caught_directly():
    with pytest.raises(XeraError):
        raise XeraError("something is wrong")


def test_xera_hook_message_is_reason_unprefixed():
    # XeraHook doesn't add its own "XeraHook: " prefix -- JAX's own
    # wrapping already prepends the exception's class name once the
    # exception is raised inside an io_callback and re-wrapped, so
    # adding a prefix here would just double it up.
    exc = XeraHook("EarlyStopping: no improvement")
    assert str(exc) == "EarlyStopping: no improvement"
    assert exc.reason == "EarlyStopping: no improvement"


def test_xera_hook_default_reason():
    exc = XeraHook()
    assert str(exc) == "XeraHook"


def test_except_xera_error_does_not_catch_xera_hook():
    with pytest.raises(XeraHook):
        try:
            raise XeraHook("stop")
        except XeraError:
            pytest.fail("XeraError should not catch XeraHook")


def test_except_xera_hook_does_not_catch_xera_error():
    with pytest.raises(XeraError):
        try:
            raise XeraError("bad config")
        except XeraHook:
            pytest.fail("XeraHook should not catch XeraError")


def test_wrapped_xera_hook_survives_as_message_text_through_io_callback():
    # Mirrors what Callback.is_training_stopped relies on: once JAX
    # re-wraps the exception raised inside io_callback, it's no longer
    # an XeraHook instance, but "XeraHook" must still be visible in the
    # wrapped exception's message.
    class AlwaysStop(Hook):
        def on_step_end(self, step, logs):
            Callback.stop("boundary check")

    hooks = [AlwaysStop()]
    loop = Loop(type="scan", steps=3)

    def step(carry, i):
        Callback.run_hooks(i, hooks, loss=jnp.asarray(0.0))
        return carry, i

    with pytest.raises(jax.errors.JaxRuntimeError) as exc_info:
        loop.run(step, init_carry=0)

    assert "XeraHook" in str(exc_info.value)
    assert "boundary check" in str(exc_info.value)
