"""Tests for xera.weave.callback: Callback registry and io_callback checkpointing."""

import jax
import jax.numpy as jnp
import pytest
import xera.loom as loom
import xera.weave as weave
from xera.weave.callback import Callback
from xera.weave.metrics import Metrics
from xera.serialize.model import load_model


@pytest.fixture(autouse=True)
def _clean_registry():
    # Callback._registry and Metrics._registry are class-level dicts shared
    # across tests; make sure each test starts and ends with a clean slate
    # so registrations don't leak between tests.
    Callback._registry.clear()
    Metrics._registry.clear()
    yield
    Callback._registry.clear()
    Metrics._registry.clear()


def test_callback_accessible_from_weave_namespace():
    assert weave.Callback is Callback


# ---------------------------------------------------------------------------
# register / unregister
# ---------------------------------------------------------------------------

def test_register_adds_to_registry_via_direct_call():
    def fn(step, value):
        pass

    Callback.register("ckpt", fn)
    assert Callback._registry["ckpt"] is fn


def test_register_as_decorator():
    @Callback.register("decorated")
    def fn(step, value):
        pass

    assert Callback._registry["decorated"] is fn


def test_register_decorator_returns_original_function():
    def original(step, value):
        return step, value

    registered = Callback.register("some_name")(original)
    assert registered is original


def test_unregister_removes_entry():
    Callback.register("temp", lambda step, value: None)
    assert "temp" in Callback._registry
    Callback.unregister("temp")
    assert "temp" not in Callback._registry


def test_unregister_missing_key_does_not_raise():
    Callback.unregister("does_not_exist")


# ---------------------------------------------------------------------------
# call() -- generic ordered io_callback dispatch
# ---------------------------------------------------------------------------

def test_call_invokes_registered_function():
    calls = []
    Callback.register("touch", lambda step, value: calls.append((int(step), value)))

    Callback.call("touch", jnp.asarray(3), 1.5)
    jax.effects_barrier()

    assert calls == [(3, 1.5)]


def test_call_unregistered_name_raises_keyerror():
    with pytest.raises(KeyError, match="no function registered"):
        Callback.call("missing", jnp.asarray(0))


def test_call_is_jit_compatible_via_io_callback():
    calls = []
    Callback.register("touch", lambda step, value: calls.append(int(value)))

    @jax.jit
    def f(x):
        Callback.call("touch", jnp.asarray(0), x)
        return x + 1

    out = f(jnp.asarray(41))
    jax.effects_barrier()
    assert int(out) == 42
    assert calls == [41]


def test_call_preserves_order_across_scan_steps():
    # This is the whole point of using io_callback(ordered=True) instead of
    # jax.debug.callback: writes triggered from inside a scan must land in
    # the same order the corresponding steps ran.
    order = []
    Callback.register("mark", lambda step, value: order.append(int(step)))

    def body(carry, i):
        def do_call():
            Callback.call("mark", i, i)
            return None

        def skip():
            return None

        jax.lax.cond(i % 2 == 0, do_call, skip)
        return carry, None

    jax.lax.scan(body, 0, jnp.arange(6))
    jax.effects_barrier()

    assert order == [0, 2, 4]


# ---------------------------------------------------------------------------
# log() -- dispatches through Metrics._registry, but via ordered io_callback
# ---------------------------------------------------------------------------

def test_log_dispatches_to_metrics_registered_fn():
    calls = []
    Metrics.register("loss", lambda step, value: calls.append((int(step), float(value))))

    Callback.log(step=5, loss=0.25)
    jax.effects_barrier()

    assert calls == [(5, 0.25)]


def test_log_shares_registry_with_metrics_log():
    # Registering once via Metrics.register should work for both dispatch
    # mechanisms -- Metrics.log (best-effort) and Callback.log (ordered).
    calls = []
    Metrics.register("loss", lambda step, value: calls.append("via_registry"))

    Metrics.log(step=1, loss=0.1)
    Callback.log(step=2, loss=0.2)
    jax.effects_barrier()

    assert calls == ["via_registry", "via_registry"]


def test_log_falls_back_to_default_emit_for_unregistered_metric(capsys):
    Callback.log(step=3, unregistered_metric=1.5)
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert "unregistered_metric" in captured.out
    assert "1.5" in captured.out


def test_log_writes_durable_file_via_registered_fn(tmp_path):
    log_path = tmp_path / "train.log"

    @Metrics.register("loss")
    def _to_file(step, value):
        with open(log_path, "a") as f:
            f.write(f"{int(step)},{float(value)}\n")

    Callback.log(step=0, loss=1.0)
    Callback.log(step=1, loss=0.5)
    jax.effects_barrier()

    lines = log_path.read_text().splitlines()
    assert lines == ["0,1.0", "1,0.5"]


def test_log_preserves_order_across_scan_steps(tmp_path):
    log_path = tmp_path / "train.log"

    @Metrics.register("loss")
    def _to_file(step, value):
        with open(log_path, "a") as f:
            f.write(f"{int(step)}\n")

    def body(carry, i):
        Callback.log(step=i, loss=i.astype(jnp.float32))
        return carry, None

    jax.lax.scan(body, 0, jnp.arange(5))
    jax.effects_barrier()

    assert log_path.read_text().splitlines() == ["0", "1", "2", "3", "4"]


def test_log_none_step_passed_to_registered_fn():
    calls = []
    Metrics.register("x", lambda step, value: calls.append(step))
    Callback.log(x=1.0)
    jax.effects_barrier()
    assert calls == [None]


# ---------------------------------------------------------------------------
# save_model -- checkpoint convenience wrapper
# ---------------------------------------------------------------------------

def test_save_model_writes_safetensors_file(tmp_path):
    model = loom.Dense(4, 8, key=jax.random.PRNGKey(0))
    path_fn = lambda step: str(tmp_path / f"ckpt_{step}.safetensors")

    Callback.save_model(jnp.asarray(7), model, path_fn)
    jax.effects_barrier()

    saved_path = tmp_path / "ckpt_7.safetensors"
    assert saved_path.exists()

    template = loom.Dense(4, 8, key=jax.random.PRNGKey(1))
    loaded = load_model(template, str(saved_path))
    assert jnp.allclose(loaded.weight, model.weight)
    assert jnp.allclose(loaded.bias, model.bias)


def test_save_model_does_not_go_through_registry():
    # save_model is a direct io_callback wrapper, not routed through
    # Callback._registry, so it works with no prior registration.
    assert "checkpoint" not in Callback._registry


def test_save_model_is_jit_compatible(tmp_path):
    model = loom.Dense(2, 2, key=jax.random.PRNGKey(0))
    path_fn = lambda step: str(tmp_path / f"jit_ckpt_{step}.safetensors")

    @jax.jit
    def f(m, i):
        Callback.save_model(i, m, path_fn)
        return i + 1

    f(model, jnp.asarray(1))
    jax.effects_barrier()
    assert (tmp_path / "jit_ckpt_1.safetensors").exists()


# ---------------------------------------------------------------------------
# save_state -- checkpoint convenience wrapper
# ---------------------------------------------------------------------------

def test_save_state_writes_safetensors_file(tmp_path):
    from xera.weave.optimizer.core.adam import Adam
    from xera.serialize.state import load_state

    opt = Adam(lr=0.1)
    params = {"w": jnp.ones((3,))}
    state = opt.init(params)
    grads = jax.tree_util.tree_map(lambda p: p * 0.5, params)
    _, state = opt.update(grads, state, params)

    path_fn = lambda step: str(tmp_path / f"opt_{step}.safetensors")
    Callback.save_state(jnp.asarray(2), state, path_fn)
    jax.effects_barrier()

    saved_path = tmp_path / "opt_2.safetensors"
    assert saved_path.exists()

    template = opt.init(params)
    loaded = load_state(template, str(saved_path))
    assert int(loaded.step) == int(state.step)
    assert jnp.allclose(loaded.m["w"], state.m["w"])
