"""Tests for xera.loom.auto_flash_attention: AutoFA backend dispatch.

There is no naive/portable kernel and no silent fallback. AutoFA supports
exactly two backends -- cuDNN (GPU) and splash attention (TPU) -- and
raises immediately for any unsupported platform, dtype, or feature
(additive bias, local windowing) rather than substituting a different
implementation. This environment has no GPU/TPU, so the real backend
calls (`_flash_attention_cudnn` / `_flash_attention_splash`) aren't
exercised end-to-end here; the preflight compatibility checks and the
dispatcher's error behavior are.
"""

import unittest.mock as mock

import jax.numpy as jnp
import pytest
import xera.loom as loom
from xera.loom.auto_flash_attention import (
    _cudnn_compatibility_issue,
    _splash_compatibility_issue,
)


def _make_qkv(batch, heads, seq_len, head_dim, dtype=jnp.bfloat16):
    return (
        jnp.zeros((batch, heads, seq_len, head_dim), dtype=dtype),
        jnp.zeros((batch, heads, seq_len, head_dim), dtype=dtype),
        jnp.zeros((batch, heads, seq_len, head_dim), dtype=dtype),
    )


# ---------------------------------------------------------------------------
# Public API surface.
# ---------------------------------------------------------------------------

def test_auto_flash_attention_exposed_on_loom():
    assert hasattr(loom, "auto_flash_attention")


def test_invalid_backend_raises_value_error():
    q, k, v = _make_qkv(1, 2, 8, 8)
    with pytest.raises(ValueError, match="backend must be one of"):
        loom.auto_flash_attention(q, k, v, backend="bogus")


def test_naive_backend_no_longer_a_valid_option():
    # "naive" used to be a valid backend value; it no longer exists.
    q, k, v = _make_qkv(1, 2, 8, 8)
    with pytest.raises(ValueError, match="backend must be one of"):
        loom.auto_flash_attention(q, k, v, backend="naive")


# ---------------------------------------------------------------------------
# Dispatcher: no fallback anywhere -- unsupported platform/dtype/feature
# raises immediately.
# ---------------------------------------------------------------------------

def test_auto_on_unsupported_platform_raises_not_implemented():
    # This test environment's real platform is CPU, which AutoFA does not
    # support at all now that there's no naive/portable kernel.
    q, k, v = _make_qkv(1, 2, 8, 8)
    with pytest.raises(NotImplementedError):
        loom.auto_flash_attention(q, k, v)


def test_forced_cudnn_backend_raises_without_gpu_support():
    # Forcing backend="cudnn" on a non-bf16/fp16 or non-GPU-compatible
    # call must raise, never fall back to anything else.
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    with pytest.raises(ValueError, match="cudnn"):
        loom.auto_flash_attention(q, k, v, backend="cudnn")


def test_forced_splash_backend_raises_on_unsupported_dtype():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    with pytest.raises(ValueError, match="splash"):
        loom.auto_flash_attention(q, k, v, backend="splash")


def test_forced_cudnn_backend_rejects_bias():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 2, 8, 8), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="bias"):
        loom.auto_flash_attention(q, k, v, bias=bias, backend="cudnn")


def test_forced_cudnn_backend_rejects_local_window():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="local_window_size"):
        loom.auto_flash_attention(q, k, v, local_window_size=4, backend="cudnn")


def test_forced_splash_backend_rejects_bias():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 2, 8, 8), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="bias"):
        loom.auto_flash_attention(q, k, v, bias=bias, backend="splash")


def test_forced_splash_backend_rejects_local_window():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="local_window_size"):
        loom.auto_flash_attention(q, k, v, local_window_size=4, backend="splash")


def test_auto_on_simulated_gpu_with_unsupported_dtype_raises():
    # Simulate a GPU platform (mocking jax.devices()) to exercise the
    # "auto" dispatch path's GPU branch without needing real GPU hardware.
    # fp32 is not supported by cuDNN, and there is no fallback -- this
    # must raise rather than silently using a different implementation.
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.float32)

    fake_device = mock.MagicMock()
    fake_device.platform = "gpu"

    with mock.patch("jax.devices", return_value=[fake_device]):
        with pytest.raises(ValueError, match="cannot run on GPU"):
            loom.auto_flash_attention(q, k, v, causal=True)


def test_auto_on_simulated_gpu_with_bias_raises():
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 2, 16, 16), dtype=jnp.bfloat16)

    fake_device = mock.MagicMock()
    fake_device.platform = "gpu"

    with mock.patch("jax.devices", return_value=[fake_device]):
        with pytest.raises(ValueError, match="bias"):
            loom.auto_flash_attention(q, k, v, bias=bias)


def test_auto_on_simulated_tpu_with_unsupported_dtype_raises():
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.float32)

    fake_device = mock.MagicMock()
    fake_device.platform = "tpu"

    with mock.patch("jax.devices", return_value=[fake_device]):
        with pytest.raises(ValueError, match="cannot run on TPU"):
            loom.auto_flash_attention(q, k, v, causal=True)


def test_auto_on_simulated_tpu_with_local_window_raises():
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.bfloat16)

    fake_device = mock.MagicMock()
    fake_device.platform = "tpu"

    with mock.patch("jax.devices", return_value=[fake_device]):
        with pytest.raises(ValueError, match="local_window_size"):
            loom.auto_flash_attention(q, k, v, local_window_size=4)


# ---------------------------------------------------------------------------
# cuDNN / splash preflight compatibility checks (unit-tested directly,
# since this environment has no GPU/TPU to exercise the real backends).
# ---------------------------------------------------------------------------

def test_cudnn_rejects_fp32():
    q_fp32 = jnp.zeros((1, 2, 8, 4), dtype=jnp.float32)
    issue = _cudnn_compatibility_issue(q_fp32, bias=None, local_window_size=None)
    assert issue is not None
    assert "float32" in issue
    assert "cuDNN" in issue


def test_cudnn_accepts_bf16_with_no_extra_features():
    q_bf16 = jnp.zeros((1, 2, 8, 4), dtype=jnp.bfloat16)
    issue = _cudnn_compatibility_issue(q_bf16, bias=None, local_window_size=None)
    assert issue is None


def test_cudnn_rejects_bias():
    q_bf16 = jnp.zeros((1, 2, 8, 4), dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 2, 8, 8))
    issue = _cudnn_compatibility_issue(q_bf16, bias=bias, local_window_size=None)
    assert issue is not None
    assert "bias" in issue


def test_cudnn_rejects_local_window():
    q_bf16 = jnp.zeros((1, 2, 8, 4), dtype=jnp.bfloat16)
    issue = _cudnn_compatibility_issue(q_bf16, bias=None, local_window_size=4)
    assert issue is not None
    assert "local_window_size" in issue


def test_splash_accepts_plain_request():
    q_bf16 = jnp.zeros((1, 2, 8, 4), dtype=jnp.bfloat16)
    issue = _splash_compatibility_issue(q_bf16, bias=None, local_window_size=None)
    assert issue is None


def test_splash_rejects_fp32():
    q_fp32 = jnp.zeros((1, 2, 8, 4), dtype=jnp.float32)
    issue = _splash_compatibility_issue(q_fp32, bias=None, local_window_size=None)
    assert issue is not None
    assert "float32" in issue


def test_splash_rejects_bias():
    q_bf16 = jnp.zeros((1, 2, 8, 4), dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 2, 8, 8))
    issue = _splash_compatibility_issue(q_bf16, bias=bias, local_window_size=None)
    assert issue is not None
    assert "bias" in issue


def test_splash_rejects_local_window():
    q_bf16 = jnp.zeros((1, 2, 8, 4), dtype=jnp.bfloat16)
    issue = _splash_compatibility_issue(q_bf16, bias=None, local_window_size=4)
    assert issue is not None
    assert "local_window_size" in issue
