"""Tests for xera.weave.loop (Loop) and xera.weave.struct (Struct).

`Struct` replaces the old `State`/`Train` pair: there is no built-in
`Train` class anymore. Instead `Train` is just a naming convention for a
plain `Struct` subclass whose fields hold other `Struct`/`Module`
instances (a dataset, an optimizer-driving step, ...) and which defines
its own `run()`. These tests cover `Struct`'s mechanics directly (dataclass
fields, `rng()`, pytree flatten/unflatten) and then exercise that pattern
end-to-end via a small `Trainer(Struct)` example, mirroring the README.
"""

import jax
import jax.numpy as jnp
import pytest
import xera.loom as loom
import xera.weave as weave
from xera.weave.loop import Loop
from xera.weave.struct import Struct
from xera.weave.optimizer.base import apply_updates
from xera.weave.optimizer.core.sgd import SGDMomentum
from xera.weave.optimizer.core.adam import Adam


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

def test_loop_default_type_is_scan():
    loop = Loop(steps=5)
    assert loop.type == "scan"


def test_loop_is_a_struct():
    assert isinstance(Loop(steps=5), Struct)


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
# Struct — base mechanics
# ---------------------------------------------------------------------------

def test_struct_accessible_from_weave_namespace():
    assert weave.Struct is Struct


def test_struct_sets_fields_from_kwargs():
    class Counter(Struct):
        step: int = 0
        total: float = 0.0

    c = Counter(step=3, total=1.5)
    assert c.step == 3
    assert c.total == 1.5


def test_struct_sets_fields_from_positional_args():
    class Point(Struct):
        x: int = 0
        y: int = 0

    p = Point(1, 2)
    assert (p.x, p.y) == (1, 2)


def test_struct_setup_hook_runs_on_init():
    class WithSetup(Struct):
        steps: int = 4

        def setup(self):
            self.loop = Loop(type="scan", steps=self.steps)

    s = WithSetup(steps=4)
    assert isinstance(s.loop, Loop)


def test_struct_rng_without_key_raises():
    class Datasets(Struct):
        x: jnp.ndarray = None

        def augment(self):
            return self.x + jax.random.normal(self.rng(), self.x.shape)

    d = Datasets(x=jnp.zeros((3,)))
    with pytest.raises(RuntimeError):
        d.augment()


def test_struct_rng_with_key_works():
    class Datasets(Struct):
        x: jnp.ndarray = None

        def augment(self):
            return self.x + jax.random.normal(self.rng(), self.x.shape)

    d = Datasets(x=jnp.zeros((3,)), key=jax.random.PRNGKey(0))
    out = d.augment()
    assert out.shape == (3,)
    assert not jnp.allclose(out, jnp.zeros((3,)))


def test_struct_rng_pool_retained_after_init():
    # Unlike Module, a Struct's RNG pool stays live for the instance's
    # lifetime when constructed with key=, since Struct methods (e.g.
    # dataset augmentation) are meant to call self.rng() on every call,
    # not just once during setup().
    class Datasets(Struct):
        x: jnp.ndarray = None

    d = Datasets(x=jnp.zeros((3,)), key=jax.random.PRNGKey(0))
    k = d.rng()
    assert k.shape == (2,)


def test_struct_repr_shows_fields():
    class Counter(Struct):
        step: int = 0

    assert repr(Counter(step=7)) == "Counter(step=7)"


# ---------------------------------------------------------------------------
# Struct — pytree flatten/unflatten (dynamic vs static)
# ---------------------------------------------------------------------------

def test_struct_ndarray_field_is_dynamic():
    class Holder(Struct):
        arr: jnp.ndarray = None
        name: str = "cfg"

    h = Holder(arr=jnp.ones((2,)), name="cfg")
    leaves = jax.tree_util.tree_leaves(h)
    assert len(leaves) == 1
    assert jnp.array_equal(leaves[0], jnp.ones((2,)))


def test_struct_nested_struct_field_is_dynamic():
    class Inner(Struct):
        v: jnp.ndarray = None

    class Outer(Struct):
        inner: Inner = None

    o = Outer(inner=Inner(v=jnp.array([1.0, 2.0])))
    leaves = jax.tree_util.tree_leaves(o)
    assert len(leaves) == 1
    assert jnp.array_equal(leaves[0], jnp.array([1.0, 2.0]))


def test_struct_module_field_is_dynamic():
    model = loom.Dense(2, 3, key=jax.random.PRNGKey(0))

    class Holder(Struct):
        model: object = None

    h = Holder(model=model)
    leaves = jax.tree_util.tree_leaves(h)
    assert len(leaves) == len(jax.tree_util.tree_leaves(model))


def test_struct_list_of_struct_field_is_dynamic():
    class Cb(Struct):
        v: jnp.ndarray = None

    class Callbacks(Struct):
        items: list = None

    c = Callbacks(items=[Cb(v=jnp.array([1.0])), Cb(v=jnp.array([2.0]))])
    leaves = jax.tree_util.tree_leaves(c)
    assert len(leaves) == 2


def test_struct_dict_of_struct_field_is_dynamic():
    class Cb(Struct):
        v: jnp.ndarray = None

    class Callbacks(Struct):
        items: dict = None

    c = Callbacks(items={"a": Cb(v=jnp.array([1.0])), "b": Cb(v=jnp.array([2.0]))})
    leaves = jax.tree_util.tree_leaves(c)
    assert len(leaves) == 2


def test_struct_plain_config_field_is_static():
    class Cfg(Struct):
        lr: float = 0.1
        arr: jnp.ndarray = None

    c = Cfg(lr=0.1, arr=jnp.ones((2,)))
    leaves = jax.tree_util.tree_leaves(c)
    # Only the array leaf; lr is static aux data, not a pytree leaf.
    assert len(leaves) == 1


def test_struct_roundtrips_through_tree_flatten_unflatten():
    class Cfg(Struct):
        lr: float = 0.1
        arr: jnp.ndarray = None

    c = Cfg(lr=0.1, arr=jnp.array([1.0, 2.0, 3.0]))
    leaves, treedef = jax.tree_util.tree_flatten(c)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.lr == 0.1
    assert jnp.array_equal(rebuilt.arr, c.arr)


def test_struct_jit_traces_through_dynamic_fields():
    class Cfg(Struct):
        arr: jnp.ndarray = None

    @jax.jit
    def double(c):
        return c.arr * 2

    out = double(Cfg(arr=jnp.array([1.0, 2.0])))
    assert jnp.array_equal(out, jnp.array([2.0, 4.0]))


def test_struct_grad_reports_readable_attribute_path():
    # Keyed pytree registration means an error inside a nested Struct
    # field surfaces the attribute path (e.g. "cfg.arr"), same as Module.
    class Cfg(Struct):
        arr: jnp.ndarray = None

    c = Cfg(arr=jnp.array([1.0, 2.0]))
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(c)
    path, _ = leaves_with_path[0]
    assert any(getattr(p, "name", None) == "arr" for p in path)


# ---------------------------------------------------------------------------
# Struct as a training driver ("Train" pattern, per the README)
# ---------------------------------------------------------------------------

def _make_linear_model():
    return loom.Dense(2, 1, use_bias=False, key=jax.random.PRNGKey(0))


class _LinearTrainer(Struct):
    """A Trainer written as a plain Struct, per the README pattern."""

    optimizer: "Optimizer" = None
    steps: int = 100

    def setup(self):
        assert self.optimizer is not None, "Trainer requires an `optimizer=`."
        self.loop = Loop(type="scan", steps=self.steps)
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

    def run(self, model):
        opt_state = self.optimizer.init(model)
        (final_model, final_opt_state), losses = self.loop.run(
            self.step, (model, opt_state)
        )
        return final_model, final_opt_state, losses


def test_trainer_struct_requires_optimizer():
    with pytest.raises(AssertionError):
        _LinearTrainer(optimizer=None)


def test_trainer_struct_default_steps_and_loop():
    trainer = _LinearTrainer(optimizer=SGDMomentum(lr=0.01))
    assert trainer.steps == 100
    assert isinstance(trainer.loop, Loop)


def test_trainer_struct_run_returns_model_state_and_losses():
    model = _make_linear_model()
    trainer = _LinearTrainer(optimizer=SGDMomentum(lr=0.01), steps=5)
    final_model, final_opt_state, losses = trainer.run(model)

    assert final_model.weight.shape == model.weight.shape
    assert losses.shape == (5,)
    assert final_opt_state is not None


def test_trainer_struct_reduces_loss_over_steps():
    model = _make_linear_model()

    def initial_loss_of(m):
        pred = m(jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]))
        target = jnp.array([[2.0], [3.0], [5.0], [7.0]])
        return float(jnp.mean((pred - target) ** 2))

    loss_before = initial_loss_of(model)

    trainer = _LinearTrainer(optimizer=Adam(lr=0.1), steps=100)
    trained_model, _, _ = trainer.run(model)

    loss_after = initial_loss_of(trained_model)
    assert loss_after < loss_before


def test_trainer_struct_step_applies_one_optimizer_update():
    model = _make_linear_model()
    trainer = _LinearTrainer(optimizer=SGDMomentum(lr=0.1, momentum=0.0), steps=1)
    opt_state = trainer.optimizer.init(model)
    (new_model, new_opt_state), loss = trainer.step((model, opt_state), 0)
    assert new_model.weight.shape == model.weight.shape
    assert not jnp.allclose(new_model.weight, model.weight)
    assert loss.shape == ()


def test_trainer_struct_fori_loop_type_runs_successfully():
    model = _make_linear_model()

    class Trainer(_LinearTrainer):
        def setup(self):
            super().setup()
            self.loop = Loop(type="fori_loop", steps=self.steps)

    trainer = Trainer(optimizer=SGDMomentum(lr=0.01), steps=3)
    final_model, _, _ = trainer.run(model)
    assert final_model.weight.shape == model.weight.shape


def test_trainer_struct_log_every_via_metrics_does_not_crash(capsys):
    from xera.weave.metrics import Metrics

    model = _make_linear_model()

    class Trainer(_LinearTrainer):
        log_every: int = 2

        def step(self, carry, i):
            (model, opt_state), loss = super().step(carry, i)
            jax.lax.cond(
                i % self.log_every == 0,
                lambda: Metrics.log(i, loss=loss),
                lambda: None,
            )
            return (model, opt_state), loss

    trainer = Trainer(optimizer=SGDMomentum(lr=0.01), steps=4, log_every=2)
    final_model, _, _ = trainer.run(model)
    jax.effects_barrier()
    assert final_model.weight.shape == model.weight.shape


def test_trainer_struct_checkpoint_every_writes_files_via_callback(tmp_path):
    from xera.weave.callback import Callback

    model = _make_linear_model()

    class Trainer(_LinearTrainer):
        checkpoint_every: int = 2

        def setup(self):
            super().setup()
            self._ckpt_path_fn = lambda step: str(tmp_path / f"ckpt_{step}.safetensors")

        def step(self, carry, i):
            (model, opt_state), loss = super().step(carry, i)
            jax.lax.cond(
                i % self.checkpoint_every == 0,
                lambda: Callback.save_model(i, model, self._ckpt_path_fn),
                lambda: None,
            )
            return (model, opt_state), loss

    trainer = Trainer(optimizer=SGDMomentum(lr=0.01), steps=4, checkpoint_every=2)
    final_model, _, _ = trainer.run(model)
    jax.effects_barrier()

    assert final_model.weight.shape == model.weight.shape
    # checkpoint_every=2 over steps 0..3 -> writes at step 0 and step 2.
    assert (tmp_path / "ckpt_0.safetensors").exists()
    assert (tmp_path / "ckpt_2.safetensors").exists()
    assert not (tmp_path / "ckpt_1.safetensors").exists()


def test_trainer_struct_durable_log_writes_via_callback(tmp_path):
    from xera.weave.metrics import Metrics
    from xera.weave.callback import Callback

    model = _make_linear_model()
    log_path = tmp_path / "train.log"

    @Metrics.register("loss")
    def _to_file(step, value):
        with open(log_path, "a") as f:
            f.write(f"{int(step)},{float(value)}\n")

    class Trainer(_LinearTrainer):
        log_every: int = 2

        def step(self, carry, i):
            (model, opt_state), loss = super().step(carry, i)
            jax.lax.cond(
                i % self.log_every == 0,
                lambda: Callback.log(i, loss=loss),
                lambda: None,
            )
            return (model, opt_state), loss

    try:
        trainer = Trainer(optimizer=SGDMomentum(lr=0.01), steps=4, log_every=2)
        trainer.run(model)
        jax.effects_barrier()

        lines = log_path.read_text().splitlines()
        # log_every=2 over steps 0..3 -> fires at step 0 and step 2.
        assert len(lines) == 2
        assert lines[0].startswith("0,")
        assert lines[1].startswith("2,")
    finally:
        Metrics.unregister("loss")
