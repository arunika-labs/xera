"""Tests for xera.weave.hook / xera.weave.early_stopping: lifecycle
callbacks and training-stop propagation."""

import jax
import jax.numpy as jnp
import pytest
import xera.weave as weave
from xera.weave import Struct
from xera.weave.hook import Hook
from xera.weave.early_stopping import EarlyStopping
from xera.weave.callback import Callback, XeraHook
from xera.weave.loop import Loop
from xera.weave.metrics import Metrics


@pytest.fixture(autouse=True)
def _clean_registry():
    Callback._registry.clear()
    Metrics._registry.clear()
    yield
    Callback._registry.clear()
    Metrics._registry.clear()


# ---------------------------------------------------------------------------
# namespace / class hierarchy
# ---------------------------------------------------------------------------

def test_hook_accessible_from_weave_namespace():
    assert weave.Hook is Hook


def test_early_stopping_accessible_from_weave_namespace():
    assert weave.EarlyStopping is EarlyStopping


def test_training_stopped_accessible_from_weave_namespace():
    assert weave.XeraHook is XeraHook


def test_hook_is_a_struct():
    assert issubclass(Hook, Struct)


def test_early_stopping_is_a_hook():
    assert issubclass(EarlyStopping, Hook)
    assert isinstance(EarlyStopping(), Struct)


def test_hook_default_on_step_end_is_noop():
    h = Hook()
    # Should not raise even with arbitrary args.
    h.on_step_end(0, {"loss": 1.0})


def test_hook_has_no_on_train_end():
    # Hook is intentionally scoped to a single lifecycle point
    # (on_step_end) for stateful stop-conditions -- not a general
    # lifecycle system.
    assert not hasattr(Hook(), "on_train_end")


# ---------------------------------------------------------------------------
# Hook subclassing / state
# ---------------------------------------------------------------------------

def test_custom_hook_can_declare_struct_fields():
    class PrintEveryN(Hook):
        every: int = 100

    h = PrintEveryN(every=50)
    assert h.every == 50
    assert isinstance(h, Struct)


def test_custom_hook_can_mutate_own_state_in_on_step_end():
    calls = []

    class Counter(Hook):
        def setup(self):
            object.__setattr__(self, "count", 0)

        def on_step_end(self, step, logs):
            object.__setattr__(self, "count", self.count + 1)
            calls.append(self.count)

    hook = Counter()
    hooks = [hook]
    # scan (not fori_loop): Loop's fori_loop mode traces the body once
    # up front to infer output shapes, so a side-effecting body runs an
    # extra time for step 0 -- scan calls the body exactly `steps` times.
    loop = Loop(type="scan", steps=5)

    def step(carry, i):
        Callback.run_hooks(i, hooks, loss=jnp.asarray(1.0))
        return carry, i

    loop.run(step, init_carry=0)
    assert calls == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Callback.run_hooks dispatch
# ---------------------------------------------------------------------------

def test_run_hooks_calls_on_step_end_on_every_hook_in_order():
    order = []

    class A(Hook):
        def on_step_end(self, step, logs):
            order.append(("A", int(step)))

    class B(Hook):
        def on_step_end(self, step, logs):
            order.append(("B", int(step)))

    hooks = [A(), B()]
    loop = Loop(type="scan", steps=3)

    def step(carry, i):
        Callback.run_hooks(i, hooks, loss=jnp.asarray(0.0))
        return carry, i

    loop.run(step, init_carry=0)
    assert order == [("A", 0), ("B", 0), ("A", 1), ("B", 1), ("A", 2), ("B", 2)]


def test_run_hooks_passes_concrete_logs_dict():
    seen = []

    class Spy(Hook):
        def on_step_end(self, step, logs):
            seen.append(dict(logs))

    hooks = [Spy()]
    loop = Loop(type="scan", steps=2)

    def step(carry, i):
        Callback.run_hooks(i, hooks, loss=jnp.asarray(1.5), acc=jnp.asarray(0.9))
        return carry, i

    loop.run(step, init_carry=0)
    assert len(seen) == 2
    assert set(seen[0].keys()) == {"loss", "acc"}
    assert float(seen[0]["loss"]) == pytest.approx(1.5)
    assert float(seen[0]["acc"]) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Callback.stop / XeraHook / is_training_stopped
# ---------------------------------------------------------------------------

def test_callback_stop_raises_training_stopped_directly():
    with pytest.raises(XeraHook) as exc_info:
        Callback.stop("manual stop")
    assert exc_info.value.reason == "manual stop"


def test_callback_stop_from_hook_propagates_out_of_scan():
    class AlwaysStop(Hook):
        def on_step_end(self, step, logs):
            Callback.stop("AlwaysStop hit")

    hooks = [AlwaysStop()]
    loop = Loop(type="fori_loop", steps=5)

    def step(carry, i):
        Callback.run_hooks(i, hooks, loss=jnp.asarray(0.0))
        return carry, i

    with pytest.raises(jax.errors.JaxRuntimeError) as exc_info:
        loop.run(step, init_carry=0)
    assert Callback.is_training_stopped(exc_info.value)


def test_callback_stop_from_hook_propagates_out_of_fori_loop():
    class AlwaysStop(Hook):
        def on_step_end(self, step, logs):
            Callback.stop("AlwaysStop hit")

    hooks = [AlwaysStop()]
    loop = Loop(type="fori_loop", steps=5)

    def step(carry, i):
        Callback.run_hooks(i, hooks, loss=jnp.asarray(0.0))
        return carry, i

    with pytest.raises(jax.errors.JaxRuntimeError) as exc_info:
        loop.run(step, init_carry=0)
    assert Callback.is_training_stopped(exc_info.value)


def test_is_training_stopped_false_for_unrelated_exception():
    class Broken(Hook):
        def on_step_end(self, step, logs):
            raise ValueError("something else went wrong")

    hooks = [Broken()]
    loop = Loop(type="fori_loop", steps=2)

    def step(carry, i):
        Callback.run_hooks(i, hooks, loss=jnp.asarray(0.0))
        return carry, i

    with pytest.raises(jax.errors.JaxRuntimeError) as exc_info:
        loop.run(step, init_carry=0)
    assert not Callback.is_training_stopped(exc_info.value)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

def test_early_stopping_default_fields():
    es = EarlyStopping()
    assert es.monitor == "loss"
    assert es.patience == 5
    assert es.mode == "min"
    assert es.min_delta == 0.0


def test_early_stopping_rejects_invalid_mode():
    with pytest.raises(AssertionError):
        EarlyStopping(mode="sideways")


def test_early_stopping_rejects_nonpositive_patience():
    with pytest.raises(AssertionError):
        EarlyStopping(patience=0)


def test_early_stopping_does_not_stop_while_improving():
    es = EarlyStopping(monitor="loss", patience=3, mode="min")
    hooks = [es]
    loop = Loop(type="scan", steps=10)

    def step(carry, i):
        loss = 10.0 - i.astype(jnp.float32) * 0.5  # strictly decreasing
        Callback.run_hooks(i, hooks, loss=loss)
        return carry, loss

    final, ys = loop.run(step, init_carry=0)
    assert ys.shape[0] == 10


def test_early_stopping_stops_after_patience_exhausted_min_mode():
    es = EarlyStopping(monitor="loss", patience=3, mode="min")
    hooks = [es]
    loop = Loop(type="scan", steps=20)

    def step(carry, i):
        # improves for the first 5 steps, then plateaus
        loss = jnp.where(i < 5, 10.0 - i.astype(jnp.float32), 5.0)
        Callback.run_hooks(i, hooks, loss=loss)
        return carry, loss

    with pytest.raises(jax.errors.JaxRuntimeError) as exc_info:
        loop.run(step, init_carry=0)
    assert Callback.is_training_stopped(exc_info.value)


def test_early_stopping_max_mode_stops_when_metric_plateaus_high_is_better():
    es = EarlyStopping(monitor="acc", patience=2, mode="max")
    hooks = [es]
    loop = Loop(type="scan", steps=20)

    def step(carry, i):
        acc = jnp.where(i < 3, i.astype(jnp.float32) * 0.1, 0.3)
        Callback.run_hooks(i, hooks, acc=acc)
        return carry, acc

    with pytest.raises(jax.errors.JaxRuntimeError) as exc_info:
        loop.run(step, init_carry=0)
    assert Callback.is_training_stopped(exc_info.value)


def test_early_stopping_ignores_steps_missing_monitor_key():
    es = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    hooks = [es]
    loop = Loop(type="scan", steps=5)

    def step(carry, i):
        # never provides "val_loss" -- hook should just no-op every step
        Callback.run_hooks(i, hooks, loss=jnp.asarray(1.0))
        return carry, i

    final, ys = loop.run(step, init_carry=0)
    assert ys.shape[0] == 5


def test_early_stopping_min_delta_counts_tiny_improvement_as_no_improvement():
    es = EarlyStopping(monitor="loss", patience=2, mode="min", min_delta=1.0)
    hooks = [es]
    loop = Loop(type="scan", steps=20)

    def step(carry, i):
        # improves by 0.01 each step -- well under min_delta=1.0
        loss = 10.0 - i.astype(jnp.float32) * 0.01
        Callback.run_hooks(i, hooks, loss=loss)
        return carry, loss

    with pytest.raises(jax.errors.JaxRuntimeError) as exc_info:
        loop.run(step, init_carry=0)
    assert Callback.is_training_stopped(exc_info.value)


def test_early_stopping_logs_via_metrics_before_stopping():
    logged = []

    @Metrics.register("early_stopping_triggered")
    def _capture(step, value):
        logged.append((int(step), float(value)))

    es = EarlyStopping(monitor="loss", patience=2, mode="min")
    hooks = [es]
    loop = Loop(type="scan", steps=20)

    def step(carry, i):
        loss = jnp.where(i < 3, 10.0 - i.astype(jnp.float32), 5.0)
        Callback.run_hooks(i, hooks, loss=loss)
        return carry, loss

    with pytest.raises(jax.errors.JaxRuntimeError):
        loop.run(step, init_carry=0)

    assert len(logged) == 1
