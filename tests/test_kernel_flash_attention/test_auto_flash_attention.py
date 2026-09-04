"""Tests for xera.functional.auto_flash_attention: AutoFA backend dispatch.

AutoFA supports three backends: cuDNN (GPU), splash attention (TPU), and
xenafl (`xera.loom.xenafl_attention`) -- a pure-jnp, dtype-/device-/
feature-agnostic implementation that always works, and is used as:

  - The *only* backend on CPU (or any platform other than GPU/TPU). No
    fallback is happening there, so `auto_flash_attention` never prints
    anything on CPU.
  - The fallback on GPU/TPU whenever the vendor backend for that platform
    (cuDNN/splash) can't serve the request (unsupported dtype, or a
    feature like bias/local_window_size that only xenafl supports). This
    prints a single plain `XeraInfo: ...` line explaining why -- not a
    `warnings.warn`, since falling back to xenafl is routine/expected,
    not a problem.

Forcing a specific backend (`backend="cudnn"`/`"splash"`/`"xenafl"`)
never falls back and never prints -- forcing means "use exactly this, or
raise."

`auto_flash_attention` is the public functional entry point and lives in
`xera.functional` (separate from `xera.loom`, which holds `Module`
layers plus the `xenafl_attention` kernel implementation itself).

This environment has no GPU/TPU, so the real cuDNN/splash backend calls
aren't exercised end-to-end here; the preflight compatibility checks and
the dispatcher's fallback/print behavior are (via a mocked `jax.devices`
to simulate GPU/TPU platforms).
"""

import io
import contextlib
import unittest.mock as mock

import jax.numpy as jnp
import pytest
import xera.loom as xl
import xera.functional as xf
from xera._kernel.flash_attention.auto_flash_attention import (
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

def test_auto_flash_attention_exposed_on_functional():
    assert hasattr(xf, "auto_flash_attention")
    assert hasattr(xl, "xenafl_attention")


def test_invalid_backend_raises_value_error():
    q, k, v = _make_qkv(1, 2, 8, 8)
    with pytest.raises(ValueError, match="backend must be None or one of"):
        xf.auto_flash_attention(q, k, v, backend="bogus")


def test_naive_backend_no_longer_a_valid_option():
    # "naive" used to be a valid backend value; xenafl replaced it.
    q, k, v = _make_qkv(1, 2, 8, 8)
    with pytest.raises(ValueError, match="backend must be None or one of"):
        xf.auto_flash_attention(q, k, v, backend="naive")


# ---------------------------------------------------------------------------
# Dispatcher: CPU always uses xenafl silently (nothing to fall back from).
# ---------------------------------------------------------------------------

def test_auto_on_cpu_uses_xenafl_and_is_correct():
    # This test environment's real platform is CPU. xenafl is the only
    # backend there, so this should just work (no exception).
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    out = xf.auto_flash_attention(q, k, v, causal=True)
    assert out.shape == q.shape


def test_auto_on_cpu_prints_nothing():
    # CPU isn't a fallback from anything -- xenafl is simply the only
    # option there, so no "XeraInfo" line should appear.
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        xf.auto_flash_attention(q, k, v, causal=True)
    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Dispatcher: forcing a backend never falls back and never prints.
# ---------------------------------------------------------------------------

def test_forced_cudnn_backend_raises_without_gpu_support():
    # Forcing backend="cudnn" on a non-bf16/fp16 call must raise, never
    # fall back to xenafl -- forcing means "use exactly this, or fail".
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    with pytest.raises(ValueError, match="cudnn"):
        xf.auto_flash_attention(q, k, v, backend="cudnn")


def test_forced_splash_backend_raises_on_unsupported_dtype():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    with pytest.raises(ValueError, match="splash"):
        xf.auto_flash_attention(q, k, v, backend="splash")


def test_forced_cudnn_backend_rejects_bias():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 2, 8, 8), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="bias"):
        xf.auto_flash_attention(q, k, v, bias=bias, backend="cudnn")


def test_forced_cudnn_backend_rejects_local_window():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="local_window_size"):
        xf.auto_flash_attention(q, k, v, local_window_size=4, backend="cudnn")


def test_forced_splash_backend_rejects_bias():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 2, 8, 8), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="bias"):
        xf.auto_flash_attention(q, k, v, bias=bias, backend="splash")


def test_forced_splash_backend_rejects_local_window():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="local_window_size"):
        xf.auto_flash_attention(q, k, v, local_window_size=4, backend="splash")


def test_forced_backend_never_prints_even_when_raising():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pytest.raises(ValueError):
            xf.auto_flash_attention(q, k, v, backend="cudnn")
    assert buf.getvalue() == ""


def test_forced_xenafl_backend_works_directly():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    out = xf.auto_flash_attention(q, k, v, causal=True, backend="xenafl")
    assert out.shape == q.shape


def test_forced_xenafl_backend_prints_nothing():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        xf.auto_flash_attention(q, k, v, causal=True, backend="xenafl")
    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Dispatcher: on simulated GPU/TPU, an incompatible dtype/feature falls
# back to xenafl and prints exactly one "XeraInfo" line explaining why.
# ---------------------------------------------------------------------------

def test_auto_on_simulated_gpu_with_unsupported_dtype_falls_back_and_prints():
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.float32)

    fake_device = mock.MagicMock()
    fake_device.platform = "gpu"

    buf = io.StringIO()
    with mock.patch("jax.devices", return_value=[fake_device]):
        with contextlib.redirect_stdout(buf):
            out = xf.auto_flash_attention(q, k, v, causal=True)

    assert out.shape == q.shape
    printed = buf.getvalue()
    assert printed.startswith("XeraInfo:")
    assert "xenafl" in printed
    assert "float32" in printed
    assert len(printed.strip().splitlines()) == 1


def test_auto_on_simulated_gpu_with_bias_falls_back_and_prints():
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 2, 16, 16), dtype=jnp.bfloat16)

    fake_device = mock.MagicMock()
    fake_device.platform = "gpu"

    buf = io.StringIO()
    with mock.patch("jax.devices", return_value=[fake_device]):
        with contextlib.redirect_stdout(buf):
            out = xf.auto_flash_attention(q, k, v, bias=bias)

    assert out.shape == q.shape
    printed = buf.getvalue()
    assert printed.startswith("XeraInfo:")
    assert "bias" in printed


def test_auto_on_simulated_tpu_with_unsupported_dtype_falls_back_and_prints():
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.float32)

    fake_device = mock.MagicMock()
    fake_device.platform = "tpu"

    buf = io.StringIO()
    with mock.patch("jax.devices", return_value=[fake_device]):
        with contextlib.redirect_stdout(buf):
            out = xf.auto_flash_attention(q, k, v, causal=True)

    assert out.shape == q.shape
    printed = buf.getvalue()
    assert printed.startswith("XeraInfo:")
    assert "float32" in printed


def test_auto_on_simulated_tpu_with_local_window_falls_back_and_prints():
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.bfloat16)

    fake_device = mock.MagicMock()
    fake_device.platform = "tpu"

    buf = io.StringIO()
    with mock.patch("jax.devices", return_value=[fake_device]):
        with contextlib.redirect_stdout(buf):
            out = xf.auto_flash_attention(q, k, v, local_window_size=4)

    assert out.shape == q.shape
    printed = buf.getvalue()
    assert printed.startswith("XeraInfo:")
    assert "local_window_size" in printed


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
