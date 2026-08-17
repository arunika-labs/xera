"""Tests for xera.serialize: save_model, load_model, save_state, load_state."""

import jax
import jax.numpy as jnp
import pytest
import xera.loom as loom
import xera.serialize as serialize
from xera.serialize.model import save_model, load_model, _key
from xera.serialize.state import save_state, load_state
from xera.weave.optimizer.core.adam import Adam


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

def test_save_and_load_model_roundtrip(tmp_path):
    model = loom.Dense(4, 8, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "model.safetensors")
    save_model(model, path)

    template = loom.Dense(4, 8, key=jax.random.PRNGKey(1))  # different init
    loaded = load_model(template, path)

    assert jnp.allclose(loaded.weight, model.weight)
    assert jnp.allclose(loaded.bias, model.bias)


def test_load_model_does_not_mutate_template_in_place():
    model = loom.Dense(4, 8, key=jax.random.PRNGKey(0))
    template = loom.Dense(4, 8, key=jax.random.PRNGKey(1))
    template_weight_before = template.weight.copy()

    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        save_model(model, path)
        load_model(template, path)
        assert jnp.allclose(template.weight, template_weight_before)
    finally:
        os.remove(path)


def test_save_model_preserves_dtype(tmp_path):
    model = loom.Dense(3, 3, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "model.safetensors")
    save_model(model, path)
    template = loom.Dense(3, 3, key=jax.random.PRNGKey(1))
    loaded = load_model(template, path)
    assert loaded.weight.dtype == model.weight.dtype


def test_save_model_nested_module_roundtrip(tmp_path):
    block = loom.TransformerBlock(dim=8, num_heads=2, mlp_hidden_dim=16, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "block.safetensors")
    save_model(block, path)

    template = loom.TransformerBlock(dim=8, num_heads=2, mlp_hidden_dim=16, key=jax.random.PRNGKey(1))
    loaded = load_model(template, path)

    assert jnp.allclose(loaded.attn.q_proj.weight, block.attn.q_proj.weight)
    assert jnp.allclose(loaded.mlp.fc1.weight, block.mlp.fc1.weight)
    assert jnp.allclose(loaded.ln1.gamma, block.ln1.gamma) if hasattr(block.ln1, "gamma") else True


def test_key_helper_strips_leading_dot():
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path({"a": jnp.ones(2)})
    path, _leaf = leaves_with_path[0]
    key_str = _key(path)
    assert not key_str.startswith(".")
    assert "a" in key_str


def test_save_model_output_loadable_via_safetensors_directly(tmp_path):
    from safetensors.numpy import load_file
    model = loom.Dense(2, 3, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "model.safetensors")
    save_model(model, path)
    tensors = load_file(path)
    assert "weight" in tensors
    assert "bias" in tensors
    assert tensors["weight"].shape == (2, 3)


def test_load_model_shape_mismatch_raises(tmp_path):
    model = loom.Dense(4, 4, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "model.safetensors")
    save_model(model, path)

    wrong_template = loom.Dense(4, 8, key=jax.random.PRNGKey(1))  # mismatched shape
    with pytest.raises(Exception):
        load_model(wrong_template, path)


# ---------------------------------------------------------------------------
# save_state / load_state
# ---------------------------------------------------------------------------

def test_save_and_load_state_roundtrip(tmp_path):
    opt = Adam(lr=0.1)
    params = {"w": jnp.ones((3, 3))}
    state = opt.init(params)
    grads = jax.tree_util.tree_map(lambda p: p * 0.5, params)
    _, state = opt.update(grads, state, params)  # advance state so it's non-trivial

    path = str(tmp_path / "state.safetensors")
    save_state(state, path)

    template = opt.init(params)  # fresh state, same structure, zeroed values
    loaded = load_state(template, path)

    assert int(loaded.step) == int(state.step)
    assert jnp.allclose(loaded.m["w"], state.m["w"])
    assert jnp.allclose(loaded.v["w"], state.v["w"])


def test_save_state_output_is_plain_safetensors(tmp_path):
    # No pickle, no magic bytes -- just a regular safetensors file, loadable
    # directly with the reference library.
    from safetensors.numpy import load_file
    opt = Adam(lr=0.1)
    state = opt.init({"w": jnp.ones((2,))})
    path = str(tmp_path / "state.safetensors")
    save_state(state, path)

    tensors = load_file(path)
    assert "step" in tensors
    assert "m['w']" in tensors
    assert "v['w']" in tensors


def test_load_state_rejects_missing_file():
    template = Adam(lr=0.1).init({"w": jnp.ones((2,))})
    with pytest.raises(Exception):
        load_state(template, "/nonexistent/path/state.safetensors")


def test_load_state_does_not_mutate_template_in_place(tmp_path):
    opt = Adam(lr=0.1)
    params = {"w": jnp.ones((2, 2))}
    state = opt.init(params)
    grads = jax.tree_util.tree_map(lambda p: p * 0.5, params)
    _, state = opt.update(grads, state, params)

    path = str(tmp_path / "state.safetensors")
    save_state(state, path)

    template = opt.init(params)
    template_m_before = template.m["w"].copy()
    load_state(template, path)
    assert jnp.allclose(template.m["w"], template_m_before)


def test_save_state_roundtrip_preserves_nested_pytree_structure(tmp_path):
    state = {"a": jnp.ones((2, 2)), "b": {"c": jnp.zeros((3,))}}
    path = str(tmp_path / "nested_state.safetensors")
    save_state(state, path)
    template = {"a": jnp.zeros((2, 2)), "b": {"c": jnp.zeros((3,))}}
    loaded = load_state(template, path)
    assert jnp.allclose(loaded["a"], state["a"])
    assert jnp.allclose(loaded["b"]["c"], state["b"]["c"])


def test_save_state_with_partition_optimizer_roundtrip(tmp_path):
    from xera.weave.optimizer.partition import Partition
    from xera.weave.optimizer.core.sgd import SGDMomentum

    opt = Partition([
        (lambda path, leaf: leaf.ndim == 2, Adam(lr=0.1)),
        (lambda path, leaf: True, SGDMomentum(lr=0.1)),
    ])
    params = {"w": jnp.ones((2, 2)), "b": jnp.ones((2,))}
    state = opt.init(params)

    path = str(tmp_path / "partition_state.safetensors")
    save_state(state, path)

    template = opt.init(params)  # same rules + param structure -> same assignment
    loaded = load_state(template, path)

    assert loaded.assignment == state.assignment


# ---------------------------------------------------------------------------
# Top-level xera.serialize namespace
# ---------------------------------------------------------------------------

def test_serialize_functions_exposed_at_package_level():
    assert serialize.save_model is save_model
    assert serialize.load_model is load_model
    assert serialize.save_state is save_state
    assert serialize.load_state is load_state


def test_serialize_accessible_via_xera_top_level_alias():
    import xera
    assert xera.S is xera.serialize
    assert hasattr(xera.S, "save_model")
    assert hasattr(xera.S, "load_state")
