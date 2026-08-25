"""Tests for xera.serialize.sxera's automatic checkpointing:
`load_struct` in directory mode (auto-discover + load, falling back to
a fresh start) and `checkpointer` (auto-save factory with built-in
every-N throttling via jax.lax.cond)."""

import glob
import os
import jax
import jax.numpy as jnp
import pytest
import xera.loom as loom
import xera.serialize as serialize
from xera.serialize.sxera import checkpointer, load_struct
from xera.serialize.model import save_model
from xera.weave.optimizer.core.adam import Adam


def _make_model(seed=0):
    return loom.Dense(3, 3, key=jax.random.PRNGKey(seed))


def _sxera_files(d):
    return sorted(glob.glob(os.path.join(str(d), "*.sxera")))


# ---------------------------------------------------------------------------
# load_struct() in directory mode -- fresh start when nothing exists
# ---------------------------------------------------------------------------

def test_load_struct_dir_mode_returns_templates_unchanged_when_dir_does_not_exist(tmp_path):
    model = _make_model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    metadata = {"step": 0}

    loaded_model, loaded_state, loaded_meta = load_struct(
        model, opt_state, metadata, str(tmp_path / "does_not_exist"),
    )
    assert loaded_model is model
    assert loaded_state is opt_state
    assert loaded_meta is metadata


def test_load_struct_dir_mode_returns_templates_unchanged_when_dir_is_empty(tmp_path):
    model = _make_model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    metadata = {"step": 0}

    loaded_model, loaded_state, loaded_meta = load_struct(model, opt_state, metadata, str(tmp_path))
    assert loaded_model is model
    assert loaded_state is opt_state
    assert loaded_meta is metadata


# ---------------------------------------------------------------------------
# load_struct() file mode still raises on a genuinely missing file
# (unchanged, pre-existing strict behavior -- must not regress)
# ---------------------------------------------------------------------------

def test_load_struct_file_mode_still_raises_on_missing_exact_file(tmp_path):
    model = _make_model()
    optimizer = Adam(lr=0.1)
    with pytest.raises(Exception):
        load_struct(model, optimizer.init(model), {"step": 0}, str(tmp_path / "nope.sxera"))


# ---------------------------------------------------------------------------
# checkpointer() -- factory returning a save() called from inside body_fn
# ---------------------------------------------------------------------------

def test_checkpointer_creates_directory_and_sxera_file(tmp_path):
    path = str(tmp_path / "run")
    save = checkpointer(path)
    model = _make_model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)

    save(model, opt_state, {"step": 0}, 0)
    jax.effects_barrier()
    assert os.path.isdir(path)
    assert len(_sxera_files(path)) == 1


def test_checkpointer_save_then_load_struct_roundtrips_model_and_optimizer(tmp_path):
    path = str(tmp_path)
    save = checkpointer(path)

    model = _make_model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    grads = jax.tree_util.tree_map(lambda p: p * 0.1, model)
    _, opt_state = optimizer.update(grads, opt_state, model)

    save(model, opt_state, {"step": 42}, 42)
    jax.effects_barrier()

    fresh_model = _make_model(seed=1)
    fresh_opt_state = optimizer.init(fresh_model)
    # `save`'s metadata always passes through io_callback, which turns every
    # leaf into a JAX array -- so the resume template must be array-shaped
    # too, or load_struct won't know to overwrite it from the checkpoint.
    loaded_model, loaded_state, loaded_meta = load_struct(
        fresh_model, fresh_opt_state, {"step": jnp.asarray(0)}, path,
    )

    assert jnp.allclose(loaded_model.weight, model.weight)
    assert jnp.allclose(loaded_state.m.weight, opt_state.m.weight)
    assert int(loaded_meta["step"]) == 42


def test_checkpointer_saves_every_call_by_default(tmp_path):
    path = str(tmp_path)
    save = checkpointer(path)  # every=1, override=True by default
    model = _make_model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)

    save(model, opt_state, {"step": 1}, 1)
    jax.effects_barrier()
    assert len(_sxera_files(path)) == 1
    save(model, opt_state, {"step": 2}, 2)
    jax.effects_barrier()
    assert len(_sxera_files(path)) == 1  # override=True -> still just one


def test_checkpointer_every_throttles_writes_via_lax_cond(tmp_path):
    path = str(tmp_path)
    save = checkpointer(path, every=5, override=False)
    model = _make_model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)

    for i in range(4):  # 0..3 -- only step 0 is a multiple of 5
        save(model, opt_state, {"step": i}, i)
    jax.effects_barrier()
    assert len(_sxera_files(path)) == 1  # only step 0 saved

    save(model, opt_state, {"step": 5}, 5)
    jax.effects_barrier()
    assert len(_sxera_files(path)) == 2  # step 0 and step 5


def test_checkpointer_every_works_inside_jit_scan(tmp_path):
    """The every-N skip is implemented with jax.lax.cond, so it must
    work when `save` is called from inside a jit-compiled scan, not
    just from eager Python loops."""
    from xera.weave.loop import loop

    path = str(tmp_path)
    save = checkpointer(path, every=3, override=False)
    model = _make_model()

    def step_fn(carry, i):
        save(model, None, {"step": i}, i)
        return carry, i

    loop(step_fn, init_carry=0, type="scan", steps=7)  # steps 0..6
    jax.effects_barrier()
    # multiples of 3 in [0, 7): 0, 3, 6 -> 3 files
    assert len(_sxera_files(path)) == 3


# ---------------------------------------------------------------------------
# override=True/False -- retention behavior
# ---------------------------------------------------------------------------

def test_override_true_keeps_only_latest_checkpoint(tmp_path):
    path = str(tmp_path)
    save = checkpointer(path, override=True)
    model = _make_model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)

    for step in (0, 10, 20):
        save(model, opt_state, {"step": step}, step)
    jax.effects_barrier()

    files = _sxera_files(path)
    assert len(files) == 1
    assert "000000000020" in files[0]


def test_override_false_keeps_every_checkpoint(tmp_path):
    path = str(tmp_path)
    save = checkpointer(path, override=False)
    model = _make_model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)

    for step in (0, 10, 20):
        save(model, opt_state, {"step": step}, step)
    jax.effects_barrier()

    assert len(_sxera_files(path)) == 3


def test_load_struct_dir_mode_picks_highest_step_regardless_of_override(tmp_path):
    path = str(tmp_path)
    save = checkpointer(path, override=False)
    optimizer = Adam(lr=0.1)

    models = {step: _make_model(seed=step) for step in (0, 10, 20)}
    for step, m in models.items():
        opt_state = optimizer.init(m)
        save(m, opt_state, {"step": step}, step)
    jax.effects_barrier()

    loaded_model, _, loaded_meta = load_struct(
        _make_model(seed=99), optimizer.init(_make_model(seed=99)), {"step": jnp.asarray(-1)}, path,
    )
    assert int(loaded_meta["step"]) == 20
    assert jnp.allclose(loaded_model.weight, models[20].weight)


def test_override_true_prunes_leftover_safetensors_too(tmp_path):
    path = str(tmp_path)
    model = _make_model()
    save_model(model, os.path.join(path, "model.safetensors"))

    save = checkpointer(path, override=True)
    optimizer = Adam(lr=0.1)
    save(model, optimizer.init(model), {"step": 1}, 1)
    jax.effects_barrier()

    assert len(_sxera_files(path)) == 1
    assert len(glob.glob(os.path.join(path, "*.safetensors"))) == 0


# ---------------------------------------------------------------------------
# .safetensors fallback -- model-only checkpoint, no .sxera present
# ---------------------------------------------------------------------------

def test_load_struct_dir_mode_falls_back_to_safetensors_when_no_sxera(tmp_path):
    path = str(tmp_path)
    model = _make_model()
    save_model(model, os.path.join(path, "model.safetensors"))

    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(_make_model(seed=1))
    metadata = {"step": 0}

    loaded_model, loaded_state, loaded_meta = load_struct(
        _make_model(seed=1), opt_state, metadata, path,
    )
    assert jnp.allclose(loaded_model.weight, model.weight)
    assert loaded_state is opt_state
    assert loaded_meta is metadata


def test_load_struct_dir_mode_prefers_sxera_over_safetensors(tmp_path):
    path = str(tmp_path)
    optimizer = Adam(lr=0.1)

    stale_model = _make_model(seed=1)
    save_model(stale_model, os.path.join(path, "model.safetensors"))

    fresh_model = _make_model(seed=2)
    save = checkpointer(path, override=False)
    save(fresh_model, optimizer.init(fresh_model), {"step": 5}, 5)
    jax.effects_barrier()

    loaded_model, _, loaded_meta = load_struct(
        _make_model(seed=3), optimizer.init(_make_model(seed=3)), {"step": jnp.asarray(0)}, path,
    )
    assert jnp.allclose(loaded_model.weight, fresh_model.weight)
    assert int(loaded_meta["step"]) == 5


# ---------------------------------------------------------------------------
# release= still works normally in directory mode
# ---------------------------------------------------------------------------

def test_load_struct_dir_mode_raises_on_structural_drift_without_release(tmp_path):
    path = str(tmp_path)
    model = _make_model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)

    from xera.core import Struct

    class Cfg(Struct):
        x: jnp.ndarray = None
        lr: float = 0.1

    saved = Cfg(x=jnp.ones((3,)), lr=0.1)
    save = checkpointer(path)
    save(model, opt_state, saved, 0)
    jax.effects_barrier()

    changed_template = Cfg(x=jnp.zeros((3,)), lr=0.2)
    with pytest.raises(ValueError):
        load_struct(model, opt_state, changed_template, path)


def test_load_struct_dir_mode_allows_structural_drift_with_release(tmp_path):
    path = str(tmp_path)
    model = _make_model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)

    from xera.core import Struct

    class Cfg(Struct):
        x: jnp.ndarray = None
        lr: float = 0.1

    saved = Cfg(x=jnp.ones((3,)), lr=0.1)
    save = checkpointer(path)
    save(model, opt_state, saved, 0)
    jax.effects_barrier()

    changed_template = Cfg(x=jnp.zeros((3,)), lr=0.2)
    _, _, loaded_meta = load_struct(model, opt_state, changed_template, path, release=True)
    assert loaded_meta.lr == 0.2
    assert jnp.allclose(loaded_meta.x, saved.x)


# ---------------------------------------------------------------------------
# Top-level xera.serialize namespace
# ---------------------------------------------------------------------------

def test_load_struct_and_checkpointer_exposed_at_package_level():
    assert serialize.load_struct is load_struct
    assert serialize.checkpointer is checkpointer


# ---------------------------------------------------------------------------
# End-to-end: load_struct() (directory mode) in setup(), checkpointer()'s
# save() called directly from inside a Struct-based Trainer's step() --
# no Callback.io needed anywhere, every-N handled internally via jax.lax.cond.
# ---------------------------------------------------------------------------

def test_full_trainer_pattern_with_auto_resume_and_auto_save(tmp_path):
    from xera.core import Struct
    from xera.weave.loop import loop
    from xera.weave.optimizer.base import apply_updates

    run_dir = str(tmp_path)

    class Trainer(Struct):
        model: "loom.Dense" = None
        optimizer: "Adam" = None
        path: str = run_dir
        steps: int = 6

        def setup(self):
            self.model, self.opt_state, self.meta = load_struct(
                self.model, self.optimizer.init(self.model),
                {"step": jnp.asarray(0)}, self.path,
            )
            self.save = checkpointer(self.path, every=2)

        def step(self, carry, i):
            model, opt_state = carry
            x = jnp.ones((2, 3))
            y = jnp.zeros((2, 3))

            def loss_only(m):
                return jnp.mean((m(x) - y) ** 2)

            loss, grads = jax.value_and_grad(loss_only)(model)
            updates, opt_state = self.optimizer.update(grads, opt_state, model, step=i)
            model = apply_updates(model, updates)
            self.save(model, opt_state, {"step": i}, i)
            return (model, opt_state), loss

        def run(self):
            (final_model, final_opt_state), losses = loop(
                self.step, (self.model, self.opt_state), type="scan", steps=self.steps,
            )
            self.final_model = final_model
            self.final_opt_state = final_opt_state

    trainer = Trainer(model=_make_model(), optimizer=Adam(lr=0.1))
    jax.effects_barrier()

    assert len(_sxera_files(run_dir)) == 1  # override=True default -> single latest

    # Simulate a fresh process picking training back up: a brand new
    # Trainer, given nothing but the same path, resumes automatically.
    resumed = Trainer(model=_make_model(seed=99), optimizer=Adam(lr=0.1), steps=0)
    assert int(resumed.meta["step"]) > 0
    assert not jnp.allclose(resumed.model.weight, _make_model(seed=99).weight)
