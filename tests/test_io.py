"""Tests for xera.io: save_model, load_model, save_state, load_state."""

import jax
import jax.numpy as jnp
import pytest
import xera.loom as xl
import xera.io as xio
import xera.optimizer as xopt
from xera.io.model import save_model, load_model, _key
from xera.io.state import save_state, load_state


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

def test_save_and_load_model_roundtrip(tmp_path):
    model = xl.Linear(4, 8, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "model.safetensors")
    save_model(model, path)

    template = xl.Linear(4, 8, key=jax.random.PRNGKey(1))  # different init
    loaded = load_model(template, path)

    assert jnp.allclose(loaded.weight, model.weight)
    assert jnp.allclose(loaded.bias, model.bias)


def test_load_model_does_not_mutate_template_in_place():
    model = xl.Linear(4, 8, key=jax.random.PRNGKey(0))
    template = xl.Linear(4, 8, key=jax.random.PRNGKey(1))
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
    model = xl.Linear(3, 3, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "model.safetensors")
    save_model(model, path)
    template = xl.Linear(3, 3, key=jax.random.PRNGKey(1))
    loaded = load_model(template, path)
    assert loaded.weight.dtype == model.weight.dtype


def test_save_model_nested_module_roundtrip(tmp_path):
    block = xl.TransformerBlock(dim=8, num_heads=2, mlp_hidden_dim=16, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "block.safetensors")
    save_model(block, path)

    template = xl.TransformerBlock(dim=8, num_heads=2, mlp_hidden_dim=16, key=jax.random.PRNGKey(1))
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
    model = xl.Linear(2, 3, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "model.safetensors")
    save_model(model, path)
    tensors = load_file(path)
    assert "weight" in tensors
    assert "bias" in tensors
    assert tensors["weight"].shape == (2, 3)


def test_load_model_shape_mismatch_raises(tmp_path):
    model = xl.Linear(4, 4, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "model.safetensors")
    save_model(model, path)

    wrong_template = xl.Linear(4, 8, key=jax.random.PRNGKey(1))  # mismatched shape
    with pytest.raises(Exception):
        load_model(wrong_template, path)


# ---------------------------------------------------------------------------
# save_state / load_state
# ---------------------------------------------------------------------------

def test_save_and_load_state_roundtrip(tmp_path):
    opt = xopt.Adam(lr=1e-3)
    params = {"w": jnp.ones((3, 3))}
    grads = {"w": jnp.ones((3, 3)) * 0.1}
    state = opt.init(params)
    _, state = opt.update(grads, state, params, None)
    _, state = opt.update(grads, state, params, None)

    path = str(tmp_path / "opt_state.safetensors")
    save_state(state, path)

    template = opt.init(params)  # fresh, un-stepped state as the shape template
    loaded = load_state(template, path)

    assert int(loaded.step) == int(state.step)
    assert jnp.allclose(loaded.m["w"], state.m["w"])
    assert jnp.allclose(loaded.v["w"], state.v["w"])


def test_load_state_hyperparameters_win_over_template_constructor_values(tmp_path):
    # A hyperparameter mid-schedule (or otherwise no longer at its
    # constructor default) must load back exactly, even if the template
    # optimizer used to build the shape was constructed with a
    # different value -- the checkpoint is the source of truth.
    opt = xopt.Schedule(lambda s: 1e-3 * (0.9 ** s))(xopt.Adam(lr=1e-3))
    params = {"w": jnp.ones((2, 2))}
    grads = {"w": jnp.ones((2, 2)) * 0.1}
    state = opt.init(params)
    for _ in range(5):
        _, state = opt.update(grads, state, params, None)

    path = str(tmp_path / "opt_state.safetensors")
    save_state(state, path)

    # Different constructor lr -- should be irrelevant after loading.
    opt2 = xopt.Schedule(lambda s: 1e-3 * (0.9 ** s))(xopt.Adam(lr=999.0))
    template = opt2.init(params)
    loaded = load_state(template, path)

    assert jnp.allclose(loaded.lr, state.lr)
    assert not jnp.allclose(loaded.lr, 999.0)


def test_save_state_wrapped_optimizer_roundtrip(tmp_path):
    opt = xopt.Clip(1.0)(xopt.WeightDecay(1e-4)(xopt.Adam(lr=1e-3)))
    params = {"w": jnp.ones((3, 3))}
    grads = {"w": jnp.ones((3, 3)) * 0.1}
    state = opt.init(params)
    for _ in range(3):
        _, state = opt.update(grads, state, params, None)

    path = str(tmp_path / "opt_state.safetensors")
    save_state(state, path)

    template = opt.init(params)
    loaded = load_state(template, path)

    assert jnp.allclose(loaded.threshold, state.threshold)
    assert jnp.allclose(loaded.inner_state.rate, state.inner_state.rate)
    inner = loaded.inner_state.inner_state
    expected_inner = state.inner_state.inner_state
    assert jnp.allclose(inner.lr, expected_inner.lr)
    assert jnp.allclose(inner.m["w"], expected_inner.m["w"])


def test_training_resumes_correctly_after_load_state(tmp_path):
    opt = xopt.Adam(lr=1e-3)
    params = {"w": jnp.ones((2, 2))}
    grads = {"w": jnp.ones((2, 2)) * 0.1}

    state_a = opt.init(params)
    params_a = params
    for _ in range(4):
        updates, state_a = opt.update(grads, state_a, params_a, None)
        params_a = xopt.apply_updates(params_a, updates)

    path = str(tmp_path / "opt_state.safetensors")
    save_state(state_a, path)

    # Simulate a fresh process: reload state and continue.
    state_b = load_state(opt.init(params), path)
    params_b = params_a
    updates_a, state_a = opt.update(grads, state_a, params_a, None)
    updates_b, state_b = opt.update(grads, state_b, params_b, None)

    assert jnp.allclose(updates_a["w"], updates_b["w"])
    assert int(state_a.step) == int(state_b.step)


# ---------------------------------------------------------------------------
# Top-level xera.io namespace
# ---------------------------------------------------------------------------

def test_io_functions_exposed_at_package_level():
    assert xio.save_model is save_model
    assert xio.load_model is load_model
    assert xio.save_state is save_state
    assert xio.load_state is load_state
