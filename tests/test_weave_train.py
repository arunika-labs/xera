"""Tests for xera.weave.loop (Loop) and xera.weave.train (Train)."""

import jax
import jax.numpy as jnp
import pytest
import xera.loom as loom
import xera.weave as weave
from xera.weave.loop import Loop
from xera.weave.train import Train
from xera.weave.optimizer.core.sgd import SGDMomentum
from xera.weave.optimizer.core.adam import Adam


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

def test_loop_default_type_is_scan():
    loop = Loop(steps=5)
    assert loop.type == "scan"


def test_loop_rejects_unknown_type():
    with pytest.raises(AssertionError):
        Loop(type="bogus", steps=5)


def test_loop_scan_basic_accumulation():
    loop = Loop(type="scan", steps=5)

    def step(carry, x):
        new_carry = carry + x
        return new_carry, new_carry

    final_carry, outputs = loop.run(step, init_carry=0, xs=jnp.array([1, 2, 3, 4, 5]))
    assert int(final_carry) == 15
    assert jnp.array_equal(outputs, jnp.array([1, 3, 6, 10, 15]))


def test_loop_fori_loop_basic_accumulation():
    loop = Loop(type="fori_loop", steps=5)

    def step(carry, x):
        new_carry = carry + x
        return new_carry, new_carry

    final_carry, outputs = loop.run(step, init_carry=0, xs=jnp.array([1, 2, 3, 4, 5]))
    assert int(final_carry) == 15
    assert jnp.array_equal(outputs, jnp.array([1, 3, 6, 10, 15]))


def test_loop_scan_and_fori_loop_produce_same_result():
    def step(carry, x):
        new_carry = carry * 2 + x
        return new_carry, carry

    xs = jnp.arange(6)
    scan_loop = Loop(type="scan", steps=6)
    fori_loop = Loop(type="fori_loop", steps=6)

    scan_carry, scan_out = scan_loop.run(step, init_carry=0, xs=xs)
    fori_carry, fori_out = fori_loop.run(step, init_carry=0, xs=xs)

    assert int(scan_carry) == int(fori_carry)
    assert jnp.array_equal(scan_out, fori_out)


def test_loop_default_xs_uses_arange_of_steps():
    loop = Loop(type="scan", steps=4)

    def step(carry, i):
        return carry, i

    _, outputs = loop.run(step, init_carry=None)
    assert jnp.array_equal(outputs, jnp.arange(4))


def test_loop_scalar_output_wrapped_correctly_in_fori():
    # fori_loop path coerces sample_output to an array even for scalars,
    # so it doesn't crash on `.shape`.
    loop = Loop(type="fori_loop", steps=3)

    def step(carry, x):
        return carry, 1.0  # python scalar output

    _, outputs = loop.run(step, init_carry=0, xs=jnp.arange(3))
    assert outputs.shape == (3,)
    assert jnp.allclose(outputs, jnp.ones(3))


def test_loop_pytree_carry_supported():
    loop = Loop(type="scan", steps=3)

    def step(carry, x):
        new_carry = {"a": carry["a"] + x, "b": carry["b"] - x}
        return new_carry, new_carry["a"]

    final_carry, _ = loop.run(step, init_carry={"a": 0, "b": 0}, xs=jnp.array([1, 2, 3]))
    assert int(final_carry["a"]) == 6
    assert int(final_carry["b"]) == -6


def test_loop_jit_compatible_scan():
    loop = Loop(type="scan", steps=4)

    def step(carry, x):
        return carry + x, carry

    run_fn = jax.jit(lambda c0, xs: loop.run(step, c0, xs))
    final_carry, _ = run_fn(0, jnp.arange(4))
    assert int(final_carry) == 6


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

class _LinearRegressionTrain(Train):
    """A minimal Train subclass fitting y = x @ W via MSE."""

    def setup(self):
        super().setup()
        # Fixed synthetic dataset, broadcast across steps via get_batch.
        self._x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        self._y = jnp.array([2.0, 3.0, 5.0, 7.0])

    def loss_fn(self, pred, target):
        return jnp.mean((pred - target) ** 2)

    def get_batch(self, i):
        return self._x, self._y


def _make_linear_model():
    return loom.Dense(2, 1, use_bias=False, key=jax.random.PRNGKey(0))


def test_train_requires_optimizer():
    with pytest.raises(AssertionError):
        Train(optimizer=None)


def test_train_rejects_invalid_loop_type():
    with pytest.raises(AssertionError):
        Train(optimizer=SGDMomentum(lr=0.01), loop_type="bogus")


def test_train_default_steps_and_loop_type():
    trainer = Train(optimizer=SGDMomentum(lr=0.01))
    assert trainer.steps == 100
    assert trainer.loop_type == "scan"
    assert isinstance(trainer.loop, Loop)


def test_train_call_returns_trained_model_with_same_structure():
    model = _make_linear_model()

    class Trainer(_LinearRegressionTrain):
        pass

    trainer = Trainer(optimizer=SGDMomentum(lr=0.05), steps=20)

    def loss_fn(pred, target):
        return jnp.mean((pred[:, 0] - target) ** 2)

    trainer.loss_fn = loss_fn
    trained_model = trainer(model)
    assert trained_model.weight.shape == model.weight.shape


def test_train_reduces_loss_over_steps():
    model = _make_linear_model()

    class Trainer(Train):
        def setup(self):
            super().setup()
            self._x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
            self._y = jnp.array([[2.0], [3.0], [5.0], [7.0]])

        def loss_fn(self, pred, target):
            return jnp.mean((pred - target) ** 2)

        def get_batch(self, i):
            return self._x, self._y

    def initial_loss_of(m):
        pred = m(jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]))
        target = jnp.array([[2.0], [3.0], [5.0], [7.0]])
        return float(jnp.mean((pred - target) ** 2))

    loss_before = initial_loss_of(model)

    trainer = Trainer(optimizer=Adam(lr=0.1), steps=100)
    trained_model = trainer(model)

    loss_after = initial_loss_of(trained_model)
    assert loss_after < loss_before


def test_train_run_returns_model_state_and_losses():
    model = _make_linear_model()

    class Trainer(Train):
        def setup(self):
            super().setup()
            self._x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
            self._y = jnp.array([[1.0], [1.0]])

        def loss_fn(self, pred, target):
            return jnp.mean((pred - target) ** 2)

        def get_batch(self, i):
            return self._x, self._y

    trainer = Trainer(optimizer=SGDMomentum(lr=0.01), steps=5)
    final_model, final_opt_state, losses = trainer.run(model)

    assert final_model.weight.shape == model.weight.shape
    assert losses.shape == (5,)
    assert final_opt_state is not None


def test_train_step_applies_one_optimizer_update():
    model = _make_linear_model()

    class Trainer(Train):
        def setup(self):
            super().setup()
            self._x = jnp.array([[1.0, 0.0]])
            self._y = jnp.array([[5.0]])

        def loss_fn(self, pred, target):
            return jnp.mean((pred - target) ** 2)

        def get_batch(self, i):
            return self._x, self._y

    trainer = Trainer(optimizer=SGDMomentum(lr=0.1, momentum=0.0), steps=1)
    opt_state = trainer.optimizer.init(model)
    (new_model, new_opt_state), loss = trainer.step((model, opt_state), 0)
    assert new_model.weight.shape == model.weight.shape
    assert not jnp.allclose(new_model.weight, model.weight)
    assert loss.shape == ()


def test_train_default_loss_fn_and_get_batch_raise_not_implemented():
    trainer = Train(optimizer=SGDMomentum(lr=0.01))
    with pytest.raises(NotImplementedError):
        trainer.loss_fn(1.0, 2.0)
    with pytest.raises(NotImplementedError):
        trainer.get_batch(0)


def test_train_fori_loop_type_runs_successfully():
    model = _make_linear_model()

    class Trainer(Train):
        def setup(self):
            super().setup()
            self._x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
            self._y = jnp.array([[1.0], [1.0]])

        def loss_fn(self, pred, target):
            return jnp.mean((pred - target) ** 2)

        def get_batch(self, i):
            return self._x, self._y

    trainer = Trainer(optimizer=SGDMomentum(lr=0.01), steps=3, loop_type="fori_loop")
    final_model = trainer(model)
    assert final_model.weight.shape == model.weight.shape


def test_train_log_every_does_not_crash(capsys):
    model = _make_linear_model()

    class Trainer(Train):
        def setup(self):
            super().setup()
            self._x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
            self._y = jnp.array([[1.0], [1.0]])

        def loss_fn(self, pred, target):
            return jnp.mean((pred - target) ** 2)

        def get_batch(self, i):
            return self._x, self._y

    trainer = Trainer(optimizer=SGDMomentum(lr=0.01), steps=4, log_every=2)
    final_model = trainer(model)
    jax.effects_barrier()
    assert final_model.weight.shape == model.weight.shape


def test_train_accessible_from_weave_namespace():
    assert weave.Train is Train
