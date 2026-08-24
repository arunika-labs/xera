"""Tests for xera.serialize.sxera (.sxera checkpoint format)."""

import jax
import jax.numpy as jnp
import pytest
import xera.serialize as serialize
import xera.loom as loom
from xera.core import Struct
from xera.serialize.model import load_model
from xera.serialize.sxera import save_struct, load_struct, extract_model
from xera.weave.optimizer.core.adam import Adam


def _model():
    return loom.Dense(2, 3, key=jax.random.PRNGKey(0))


# ---------------------------------------------------------------------------
# save_struct / load_struct — basic round-trip
# ---------------------------------------------------------------------------

def test_save_struct_roundtrips_model_optimizer_and_metadata(tmp_path):
    model = _model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    metadata = {"step": 42, "key": jax.random.PRNGKey(7)}
    path = str(tmp_path / "ckpt.sxera")

    save_struct(model, opt_state, metadata, path)

    loaded_model, loaded_opt_state, loaded_meta = load_struct(
        _model(), optimizer.init(_model()), {"step": 0, "key": jax.random.PRNGKey(0)}, path,
    )

    assert jnp.allclose(loaded_model.weight, model.weight)
    assert jnp.allclose(loaded_model.bias, model.bias)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.allclose, loaded_opt_state, opt_state)
    )
    assert loaded_meta["step"] == 42
    assert jnp.array_equal(loaded_meta["key"], metadata["key"])


def test_save_struct_output_is_plain_safetensors_file(tmp_path):
    from safetensors.numpy import load_file

    model = _model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    path = str(tmp_path / "ckpt.sxera")

    save_struct(model, opt_state, {"step": 1}, path)

    tensors = load_file(path)  # plain safetensors reader, no xera involved
    assert any(k.startswith("model.") for k in tensors)
    assert any(k.startswith("optimizer.") for k in tensors)


def test_save_struct_stamps_three_treedefs_in_metadata(tmp_path):
    from safetensors import safe_open

    model = _model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    path = str(tmp_path / "ckpt.sxera")
    save_struct(model, opt_state, {"step": 1}, path)

    with safe_open(path, framework="numpy") as f:
        meta = f.metadata()
    assert "xera_sxera_model_treedef" in meta
    assert "xera_sxera_optimizer_treedef" in meta
    assert "xera_sxera_metadata_treedef" in meta
    assert "xera_sxera_static_json" in meta


# ---------------------------------------------------------------------------
# metadata: mixed array / non-array leaves
# ---------------------------------------------------------------------------

def test_metadata_non_array_values_survive_roundtrip(tmp_path):
    model = _model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    path = str(tmp_path / "ckpt.sxera")

    save_struct(model, opt_state, {"step": 123, "name": "run-1"}, path)
    _, _, meta = load_struct(_model(), optimizer.init(_model()), {"step": 0, "name": ""}, path)

    assert meta["step"] == 123
    assert meta["name"] == "run-1"


def test_metadata_none_leaf_survives_roundtrip(tmp_path):
    model = _model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    path = str(tmp_path / "ckpt.sxera")

    save_struct(model, opt_state, {"step": 1, "notes": None}, path)
    _, _, meta = load_struct(_model(), optimizer.init(_model()), {"step": 0, "notes": None}, path)

    assert meta["notes"] is None


# ---------------------------------------------------------------------------
# drift detection / release=True
# ---------------------------------------------------------------------------

def test_load_struct_drift_raises_by_default(tmp_path):
    class Cfg(Struct):
        x: jnp.ndarray = None
        lr: float = 0.1

    model = _model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    path = str(tmp_path / "ckpt.sxera")
    save_struct(model, opt_state, Cfg(x=jnp.ones((2,)), lr=0.1), path)

    changed_template = Cfg(x=jnp.zeros((2,)), lr=0.2)  # hyperparameter changed
    with pytest.raises(ValueError):
        load_struct(_model(), optimizer.init(_model()), changed_template, path)


def test_load_struct_drift_allowed_with_release(tmp_path):
    class Cfg(Struct):
        x: jnp.ndarray = None
        lr: float = 0.1

    model = _model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    path = str(tmp_path / "ckpt.sxera")
    save_struct(model, opt_state, Cfg(x=jnp.ones((2,)), lr=0.1), path)

    changed_template = Cfg(x=jnp.zeros((2,)), lr=0.2)  # intentional change
    _, _, meta = load_struct(
        _model(), optimizer.init(_model()), changed_template, path, release=True,
    )
    assert meta.lr == 0.2  # template's new config wins
    assert jnp.allclose(meta.x, jnp.ones((2,)))  # array value still restored from disk


def test_load_struct_optimizer_hparam_change_with_release(tmp_path):
    model = _model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    path = str(tmp_path / "ckpt.sxera")
    save_struct(model, opt_state, {"step": 10}, path)

    new_optimizer = Adam(lr=5e-4)  # different hyperparameter than saved
    loaded_model, loaded_opt_state, meta = load_struct(
        _model(), new_optimizer.init(_model()), {"step": 0}, path, release=True,
    )
    assert meta["step"] == 10
    assert jnp.allclose(loaded_model.weight, model.weight)


def test_load_struct_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        load_struct(_model(), Adam(lr=0.1).init(_model()), {"step": 0}, str(tmp_path / "nope.sxera"))


# ---------------------------------------------------------------------------
# extract_model
# ---------------------------------------------------------------------------

def test_extract_model_produces_loadable_model_safetensors(tmp_path):
    model = _model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    sxera_path = str(tmp_path / "ckpt.sxera")
    model_path = str(tmp_path / "model.safetensors")

    save_struct(model, opt_state, {"step": 1}, sxera_path)
    extract_model(sxera_path, model_path)

    extracted = load_model(_model(), model_path)
    assert jnp.allclose(extracted.weight, model.weight)
    assert jnp.allclose(extracted.bias, model.bias)


def test_extract_model_output_has_no_prefix(tmp_path):
    from safetensors.numpy import load_file

    model = _model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    sxera_path = str(tmp_path / "ckpt.sxera")
    model_path = str(tmp_path / "model.safetensors")

    save_struct(model, opt_state, {"step": 1}, sxera_path)
    extract_model(sxera_path, model_path)

    tensors = load_file(model_path)
    assert all(not k.startswith("model.") for k in tensors)
    assert "weight" in tensors


def test_extract_model_does_not_include_optimizer_or_metadata_keys(tmp_path):
    from safetensors.numpy import load_file

    model = _model()
    optimizer = Adam(lr=0.1)
    opt_state = optimizer.init(model)
    sxera_path = str(tmp_path / "ckpt.sxera")
    model_path = str(tmp_path / "model.safetensors")

    save_struct(model, opt_state, {"step": 1, "key": jax.random.PRNGKey(0)}, sxera_path)
    extract_model(sxera_path, model_path)

    tensors = load_file(model_path)
    assert set(tensors.keys()) == {"weight", "bias"}


# ---------------------------------------------------------------------------
# package-level exposure
# ---------------------------------------------------------------------------

def test_sxera_functions_exposed_at_package_level():
    assert serialize.save_struct is save_struct
    assert serialize.load_struct is load_struct
    assert serialize.extract_model is extract_model
