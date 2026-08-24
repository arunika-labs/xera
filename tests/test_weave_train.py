"""Tests for xera.weave.loop (the `loop` function), xera.weave.callback
(`Callback`), and xera.core.Struct as a training driver.

`Struct` is the base for training-side components. A `Trainer` is not a
built-in class -- it's just a `Struct` subclass with a `run()` method:
`Struct.__init__` calls `setup()` and then, if the subclass overrides
`run`, calls `run()` too, so `Trainer(key=..., ...)` alone starts
training. `loop` itself is a plain function (not a `Struct`), composed
into `run()` however the subclass likes; `Callback.print`/`Callback.io`
are side-effects called from inside a step; `Callback.early_stopping`/
`Callback.nan` build `stop_fn`s passed as `loop(..., stop=...)`.
"""

import jax
import jax.numpy as jnp
import pytest
import xera.loom as loom
from xera.weave.loop import loop
from xera.weave.callback import Callback
from xera.core import Struct
from xera.weave.optimizer.base import apply_updates
from xera.weave.optimizer.core.sgd import SGDMomentum
from xera.weave.optimizer.core.adam import Adam


# ---------------------------------------------------------------------------
# loop() — plain function, no stop=
# ---------------------------------------------------------------------------

def test_loop_rejects_unknown_type():
    with pytest.raises(AssertionError):
        loop(lambda c, x: (c, x), init_carry=0, type="bogus", steps=5)


def test_loop_scan_basic_accumulation():
    def step(carry, x):
        new_carry = carry + x
        return new_carry, new_carry

    final_carry, outputs = loop(step, init_carry=0, xs=jnp.array([1, 2, 3, 4, 5]), type="scan", steps=5)
    assert int(final_carry) == 15
    assert jnp.array_equal(outputs, jnp.array([1, 3, 6, 10, 15]))


def test_loop_fori_loop_basic_accumulation():
    def step(carry, x):
        new_carry = carry + x
        return new_carry, new_carry

    final_carry, outputs = loop(step, init_carry=0, xs=jnp.array([1, 2, 3, 4, 5]), type="fori_loop", steps=5)
    assert int(final_carry) == 15
    assert jnp.array_equal(outputs, jnp.array([1, 3, 6, 10, 15]))


def test_loop_scan_and_fori_loop_produce_same_result():
    def step(carry, x):
        new_carry = carry * 2 + x
        return new_carry, carry

    xs = jnp.arange(6)
    scan_carry, scan_out = loop(step, init_carry=0, xs=xs, type="scan", steps=6)
    fori_carry, fori_out = loop(step, init_carry=0, xs=xs, type="fori_loop", steps=6)

    assert int(scan_carry) == int(fori_carry)
    assert jnp.array_equal(scan_out, fori_out)


def test_loop_default_xs_uses_arange_of_steps():
    def step(carry, i):
        return carry, i

    _, outputs = loop(step, init_carry=None, type="scan", steps=4)
    assert jnp.array_equal(outputs, jnp.arange(4))


def test_loop_scalar_output_wrapped_correctly_in_fori():
    # fori_loop path coerces sample_output to an array even for scalars,
    # so it doesn't crash on `.shape`.
    def step(carry, x):
        return carry, 1.0  # python scalar output

    _, outputs = loop(step, init_carry=0, xs=jnp.arange(3), type="fori_loop", steps=3)
    assert outputs.shape == (3,)
    assert jnp.allclose(outputs, jnp.ones(3))


def test_loop_pytree_carry_supported():
    def step(carry, x):
        new_carry = {"a": carry["a"] + x, "b": carry["b"] - x}
        return new_carry, new_carry["a"]

    final_carry, _ = loop(step, init_carry={"a": 0, "b": 0}, xs=jnp.array([1, 2, 3]), type="scan", steps=3)
    assert int(final_carry["a"]) == 6
    assert int(final_carry["b"]) == -6


def test_loop_jit_compatible_scan():
    def step(carry, x):
        return carry + x, carry

    run_fn = jax.jit(lambda c0, xs: loop(step, c0, xs, type="scan", steps=4))
    final_carry, _ = run_fn(0, jnp.arange(4))
    assert int(final_carry) == 6


# ---------------------------------------------------------------------------
# loop() — stop= (two-branch: real step vs. cheap dummy)
# ---------------------------------------------------------------------------

def test_loop_stop_none_runs_body_fn_every_step():
    def step(carry, x):
        return carry + x, x  # output records x, so we can check every step ran

    final_carry, outputs = loop(step, init_carry=0, xs=jnp.arange(5), type="scan", steps=5, stop=None)
    assert int(final_carry) == 10
    assert jnp.array_equal(outputs, jnp.arange(5))


def test_loop_stop_true_from_start_freezes_carry():
    def step(carry, x):
        return carry + x, carry

    final_carry, outputs = loop(
        step, init_carry=0, xs=jnp.arange(5), type="scan", steps=5,
        stop=lambda carry, x: True,
    )
    # stop fires on step 0 already -> carry never advances past init.
    assert int(final_carry) == 0
    assert jnp.array_equal(outputs, jnp.zeros(5))


def test_loop_stop_latches_after_condition_met():
    # Real accumulation up to x==2, then dummy branch for the rest.
    def step(carry, x):
        return carry + x, carry + x

    final_carry, outputs = loop(
        step, init_carry=0, xs=jnp.arange(5), type="scan", steps=5,
        stop=lambda carry, x: x >= 2,
    )
    # steps: x=0 -> carry=0 (not yet stopped when checked... see below)
    # stop(carry, x) is checked before the step runs, using x -- so the
    # step where x==2 itself already takes the dummy branch.
    assert int(final_carry) == 1  # 0 + 1, then frozen
    assert jnp.array_equal(outputs, jnp.array([0, 1, 0, 0, 0]))


def test_loop_stop_never_true_matches_stop_none():
    def step(carry, x):
        return carry + x, carry

    no_stop_carry, no_stop_out = loop(step, init_carry=0, xs=jnp.arange(5), type="scan", steps=5)
    stop_carry, stop_out = loop(
        step, init_carry=0, xs=jnp.arange(5), type="scan", steps=5,
        stop=lambda carry, x: False,
    )
    assert int(no_stop_carry) == int(stop_carry)
    assert jnp.array_equal(no_stop_out, stop_out)


def test_loop_stop_works_under_jit():
    def step(carry, x):
        return carry + x, carry

    run_fn = jax.jit(
        lambda c0, xs: loop(step, c0, xs, type="scan", steps=5, stop=lambda c, x: x >= 3)
    )
    final_carry, _ = run_fn(0, jnp.arange(5))
    assert int(final_carry) == 3  # 0+1+2, frozen once x==3 triggers stop


def test_loop_stop_with_pytree_carry():
    def step(carry, x):
        return {"a": carry["a"] + x}, carry["a"]

    final_carry, outputs = loop(
        step, init_carry={"a": 0}, xs=jnp.arange(4), type="scan", steps=4,
        stop=lambda carry, x: carry["a"] >= 3,
    )
    # stop(carry, x) is checked with the *pre-step* carry:
    #   x=0: stop({a:0},0)=False -> a=0+0=0, out=0
    #   x=1: stop({a:0},1)=False -> a=0+1=1, out=0
    #   x=2: stop({a:1},2)=False -> a=1+2=3, out=1
    #   x=3: stop({a:3},3)=True  -> dummy: a stays 3, out=0
    assert int(final_carry["a"]) == 3
    assert jnp.array_equal(outputs, jnp.array([0, 0, 1, 0]))


# ---------------------------------------------------------------------------
# Callback — print / io (side-effects, called from inside a step)
# ---------------------------------------------------------------------------

def test_callback_print_does_not_crash_inside_loop(capsys):
    def step(carry, i):
        Callback.print(i, carry=carry)
        return carry + 1, carry

    final_carry, _ = loop(step, init_carry=0, type="scan", steps=3)
    jax.effects_barrier()
    assert int(final_carry) == 3


def test_callback_io_runs_python_side_effect(tmp_path):
    log_path = tmp_path / "io.log"

    def write_line(step, value):
        with open(log_path, "a") as f:
            f.write(f"{int(step)},{float(value)}\n")

    def step(carry, i):
        Callback.io(i, write_line, carry)
        return carry + 1.0, carry

    loop(step, init_carry=0.0, type="scan", steps=3)
    jax.effects_barrier()

    lines = log_path.read_text().splitlines()
    assert lines == ["0,0.0", "1,1.0", "2,2.0"]


# ---------------------------------------------------------------------------
# Callback — early_stopping / nan (stop-condition factories)
# ---------------------------------------------------------------------------

def test_callback_early_stopping_stops_after_patience():
    # carry = (value, since_improved)
    def step(carry, x):
        value, since_improved = carry
        improved = x < 2  # "improves" for the first couple of steps
        new_since = jnp.where(improved, 0, since_improved + 1)
        return (value + x, new_since), value

    stop = Callback.early_stopping(patience=2, extract=lambda carry: carry[1])
    (final_value, _), outputs = loop(
        step, init_carry=(0, 0), xs=jnp.arange(6), type="scan", steps=6, stop=stop,
    )
    # since_improved reaches 2 once two consecutive non-improving steps
    # have happened; from then on the dummy branch freezes `value`.
    assert int(final_value) <= 1 + 2 + 3  # stopped before accumulating all of 0..5


def test_callback_nan_stops_loop():
    def step(carry, x):
        # Deliberately produce NaN once x >= 2.
        new_carry = jnp.where(x >= 2, jnp.nan, carry + x)
        return new_carry, new_carry

    final_carry, outputs = loop(
        step, init_carry=jnp.asarray(0.0), xs=jnp.arange(5), type="scan", steps=5,
        stop=Callback.nan(),
    )
    # stop(carry, x) is checked against the *pre-step* carry, same as any
    # periodic check (e.g. early stopping checked every N steps) -- so a
    # NaN produced at step N is only detected at step N+1, one step after
    # it first appears. That one NaN reaches `outputs`, but the loop
    # latches immediately after and the carry never carries NaN forward
    # past that.
    assert jnp.isnan(final_carry)  # the single NaN step's carry is what froze
    assert int(jnp.sum(jnp.isnan(outputs))) == 1


# ---------------------------------------------------------------------------
# Struct as a training driver ("Trainer" pattern) — auto-run via __init__
# ---------------------------------------------------------------------------

def _make_linear_model():
    return loom.Dense(2, 1, use_bias=False, key=jax.random.PRNGKey(0))


class _LinearTrainer(Struct):
    """A Trainer written as a plain Struct: instantiating it runs training."""

    model: "loom.Dense" = None
    optimizer: "Optimizer" = None
    steps: int = 100

    def setup(self):
        assert self.model is not None, "Trainer requires a `model=`."
        assert self.optimizer is not None, "Trainer requires an `optimizer=`."
        self._x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        self._y = jnp.array([[2.0], [3.0], [5.0], [7.0]])

    def loss_fn(self, pred, target):
        return jnp.mean((pred - target) ** 2)

    def get_batch(self, i):
        return self._x, self._y

    def step(self, carry, i):
        model, opt_state = carry
        x, y = self.get_batch(i)

        def loss_only(m):
            return self.loss_fn(m(x), y)

        loss, grads = jax.value_and_grad(loss_only)(model)
        updates, opt_state = self.optimizer.update(grads, opt_state, model, step=i)
        model = apply_updates(model, updates)
        return (model, opt_state), loss

    def run(self):
        opt_state = self.optimizer.init(self.model)
        (final_model, final_opt_state), losses = loop(
            self.step, (self.model, opt_state), type="scan", steps=self.steps,
        )
        self.final_model = final_model
        self.final_opt_state = final_opt_state
        self.losses = losses


def test_trainer_struct_requires_model_and_optimizer():
    with pytest.raises(AssertionError):
        _LinearTrainer(model=None, optimizer=SGDMomentum(lr=0.01))
    with pytest.raises(AssertionError):
        _LinearTrainer(model=_make_linear_model(), optimizer=None)


def test_trainer_struct_runs_automatically_on_construction():
    # No separate `.run()` call: instantiating Trainer(...) trains.
    trainer = _LinearTrainer(model=_make_linear_model(), optimizer=SGDMomentum(lr=0.01), steps=5)
    assert trainer.final_model.weight.shape == (2, 1)
    assert trainer.losses.shape == (5,)
    assert trainer.final_opt_state is not None


def test_trainer_struct_reduces_loss_over_steps():
    model = _make_linear_model()

    def initial_loss_of(m):
        pred = m(jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]))
        target = jnp.array([[2.0], [3.0], [5.0], [7.0]])
        return float(jnp.mean((pred - target) ** 2))

    loss_before = initial_loss_of(model)

    trainer = _LinearTrainer(model=model, optimizer=Adam(lr=0.1), steps=100)
    loss_after = initial_loss_of(trainer.final_model)
    assert loss_after < loss_before


def test_trainer_struct_step_applies_one_optimizer_update():
    model = _make_linear_model()
    optimizer = SGDMomentum(lr=0.1, momentum=0.0)
    opt_state = optimizer.init(model)

    class NoAutoRun(_LinearTrainer):
        def run(self):
            pass  # override auto-run so we can call .step() manually below

    trainer = NoAutoRun(model=model, optimizer=optimizer, steps=1)
    (new_model, new_opt_state), loss = trainer.step((model, opt_state), 0)
    assert new_model.weight.shape == model.weight.shape
    assert not jnp.allclose(new_model.weight, model.weight)
    assert loss.shape == ()


def test_trainer_struct_fori_loop_type_runs_successfully():
    class Trainer(_LinearTrainer):
        def run(self):
            opt_state = self.optimizer.init(self.model)
            (final_model, final_opt_state), losses = loop(
                self.step, (self.model, opt_state), type="fori_loop", steps=self.steps,
            )
            self.final_model = final_model

    trainer = Trainer(model=_make_linear_model(), optimizer=SGDMomentum(lr=0.01), steps=3)
    assert trainer.final_model.weight.shape == (2, 1)


# ---------------------------------------------------------------------------
# Struct — save_struct / .sxera round-trip via the Trainer pattern
# ---------------------------------------------------------------------------

def test_trainer_struct_save_struct_writes_sxera_checkpoint(tmp_path):
    trainer = _LinearTrainer(model=_make_linear_model(), optimizer=Adam(lr=0.1), steps=5)
    path = tmp_path / "ckpt.sxera"

    trainer.save_struct(
        trainer.final_model, trainer.final_opt_state,
        metadata={"step": 5}, path=str(path),
    )
    assert path.exists()


def test_trainer_struct_save_struct_roundtrips_via_load_struct(tmp_path):
    from xera.serialize.sxera import load_struct

    model = _make_linear_model()
    optimizer = Adam(lr=0.1)
    trainer = _LinearTrainer(model=model, optimizer=optimizer, steps=5)
    path = tmp_path / "ckpt.sxera"

    trainer.save_struct(
        trainer.final_model, trainer.final_opt_state,
        metadata={"step": 5}, path=str(path),
    )

    loaded_model, loaded_opt_state, loaded_meta = load_struct(
        _make_linear_model(), optimizer.init(model), {"step": 0}, str(path),
    )
    assert jnp.allclose(loaded_model.weight, trainer.final_model.weight)
    assert loaded_meta["step"] == 5
