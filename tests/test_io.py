"""Tests for xera.io: save_model, load_model."""

import jax
import jax.numpy as jnp
import pytest
import xera.loom as xl
import xera.io as xio
from xera.io.model import save_model, load_model, _key


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

def test_save_and_load_model_roundtrip(tmp_path):
    model = xl.Dense(4, 8, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "model.safetensors")
    save_model(model, path)

    template = xl.Dense(4, 8, key=jax.random.PRNGKey(1))  # different init
    loaded = load_model(template, path)

    assert jnp.allclose(loaded.weight, model.weight)
    assert jnp.allclose(loaded.bias, model.bias)


def test_load_model_does_not_mutate_template_in_place():
    model = xl.Dense(4, 8, key=jax.random.PRNGKey(0))
    template = xl.Dense(4, 8, key=jax.random.PRNGKey(1))
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
    model = xl.Dense(3, 3, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "model.safetensors")
    save_model(model, path)
    template = xl.Dense(3, 3, key=jax.random.PRNGKey(1))
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
    model = xl.Dense(2, 3, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "model.safetensors")
    save_model(model, path)
    tensors = load_file(path)
    assert "weight" in tensors
    assert "bias" in tensors
    assert tensors["weight"].shape == (2, 3)


def test_load_model_shape_mismatch_raises(tmp_path):
    model = xl.Dense(4, 4, key=jax.random.PRNGKey(0))
    path = str(tmp_path / "model.safetensors")
    save_model(model, path)

    wrong_template = xl.Dense(4, 8, key=jax.random.PRNGKey(1))  # mismatched shape
    with pytest.raises(Exception):
        load_model(wrong_template, path)


# ---------------------------------------------------------------------------
# Top-level xera.io namespace
# ---------------------------------------------------------------------------

def test_io_functions_exposed_at_package_level():
    assert xio.save_model is save_model
    assert xio.load_model is load_model
