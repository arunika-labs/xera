"""Tests for xera.weave.metrics: Metrics registry and logging."""

import jax
import pytest
import xera.weave as weave
from xera.weave.metrics import Metrics


@pytest.fixture(autouse=True)
def _clean_registry():
    # Metrics._registry is a class-level dict shared across tests; make sure
    # each test starts and ends with a clean slate so registrations don't
    # leak between tests.
    Metrics._registry.clear()
    yield
    Metrics._registry.clear()


def test_metrics_accessible_from_weave_namespace():
    assert weave.Metrics is Metrics


def test_register_adds_to_registry_via_direct_call():
    calls = []

    def logger(step, value):
        calls.append((step, value))

    Metrics.register("my_metric", logger)
    assert Metrics._registry["my_metric"] is logger


def test_register_as_decorator():
    calls = []

    @Metrics.register("decorated_metric")
    def logger(step, value):
        calls.append((step, value))

    assert Metrics._registry["decorated_metric"] is logger


def test_register_decorator_returns_original_function():
    def original(step, value):
        return step, value

    registered = Metrics.register("some_name")(original)
    assert registered is original


def test_unregister_removes_entry():
    Metrics.register("temp_metric", lambda step, value: None)
    assert "temp_metric" in Metrics._registry
    Metrics.unregister("temp_metric")
    assert "temp_metric" not in Metrics._registry


def test_unregister_missing_key_does_not_raise():
    # Should be a no-op, not an error, for a name that was never registered.
    Metrics.unregister("does_not_exist")


def test_log_calls_registered_emit_function():
    calls = []

    def logger(step, value):
        calls.append((step, value))

    Metrics.register("loss", logger)
    Metrics.log(step=5, loss=0.25)
    jax.effects_barrier()
    assert calls == [(5, 0.25)]


def test_log_multiple_metrics_dispatches_each_to_its_registered_fn():
    loss_calls = []
    acc_calls = []

    Metrics.register("loss", lambda step, value: loss_calls.append((step, value)))
    Metrics.register("accuracy", lambda step, value: acc_calls.append((step, value)))

    Metrics.log(step=1, loss=0.5, accuracy=0.9)
    jax.effects_barrier()

    assert loss_calls == [(1, 0.5)]
    assert acc_calls == [(1, 0.9)]


def test_log_unregistered_metric_falls_back_to_default_emit(capsys):
    Metrics.log(step=3, unregistered_metric=1.5)
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert "unregistered_metric" in captured.out
    assert "1.5" in captured.out


def test_log_without_step_uses_default_emit_format(capsys):
    Metrics.log(unregistered_metric=2.0)
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert "step" not in captured.out
    assert "unregistered_metric" in captured.out


def test_log_with_step_includes_step_in_default_output(capsys):
    Metrics.log(step=7, unregistered_metric=1.0)
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert "step 7" in captured.out


def test_log_none_step_passed_to_registered_fn():
    calls = []
    Metrics.register("x", lambda step, value: calls.append(step))
    Metrics.log(x=1.0)
    jax.effects_barrier()
    assert calls == [None]


def test_log_is_jit_compatible_via_debug_callback():
    # Metrics.log uses jax.debug.callback, so calling it from inside a
    # jitted function should not raise, and side effects fire once traced.
    calls = []
    Metrics.register("loss", lambda step, value: calls.append((step, value)))

    @jax.jit
    def train_step(loss_value):
        Metrics.log(step=0, loss=loss_value)
        return loss_value

    out = train_step(0.42)
    jax.effects_barrier()
    assert float(out) == pytest.approx(0.42)
    assert len(calls) == 1
