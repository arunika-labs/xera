"""Tests for xera.functional.flash_sdpa: AutoFA backend dispatch.

flash_sdpa supports three backends: cuDNN (GPU), splash attention (TPU),
and "jax" (a private pure-jnp, dtype-/device-/
feature-agnostic implementation that always works, and is used as:

  - The *only* backend on CPU (or any platform other than GPU/TPU).
  - The fallback on GPU/TPU whenever the vendor backend for that platform
    (cuDNN/splash) can't serve the request:
      * cuDNN requires an Ampere-or-newer GPU (sm_80+, checked via the
        device's `compute_capability`) and bf16/fp16 inputs. Additive bias
        and local_window_size ARE supported through AutoFA's cuDNN path --
        jax.nn.dot_product_attention's cudnn implementation forwards both
        straight into cuDNN's fused kernel -- so requesting them does not
        trigger a fallback on GPU.
      * splash requires bf16 inputs and head_dim a multiple of 128 (its
        Pallas kernel's tiling requirement), and does NOT support bias or
        local_window_size -- its kernel has no additive-bias input and no
        windowed-masking support, so requesting either on TPU falls back
        to "jax".

By default (`verbose=False`), `flash_sdpa` never prints anything --
whether `backend="auto"` fell back to "jax" or not is not reported.
Passing `verbose=True` makes a fallback print a single plain
`XeraInfo: ...` line explaining why. This is never a `warnings.warn` --
falling back to "jax" is routine/expected, not a problem -- and forcing a
specific backend (`backend="cudnn"`/`"splash"`/`"jax"`) never falls back
and never prints regardless of `verbose`, since forcing means "use
exactly this, or raise."

`flash_sdpa` is the public functional entry point and lives in
`xera.functional` (separate from `xera.loom`, which holds only `Module`
layers -- the "jax" backend's kernel implementation is private to
`xera._kernel`, not reachable from `xera.loom` or anywhere else).

This environment has no GPU/TPU, so the real cuDNN/splash backend calls
aren't exercised end-to-end here; the preflight compatibility checks and
the dispatcher's fallback/print behavior are (via a mocked `jax.devices`
to simulate GPU/TPU platforms and GPU compute capabilities).
"""

import io
import contextlib
import unittest.mock as mock

import jax.numpy as jnp
import pytest
import xera.loom as xl
import xera.functional as xf
from xera._kernel.flash_attention.flash_sdpa import (
    _cudnn_compatibility_issue,
    _splash_compatibility_issue,
    _parse_compute_capability,
)


def _make_qkv(batch, heads, seq_len, head_dim, dtype=jnp.bfloat16):
    return (
        jnp.zeros((batch, heads, seq_len, head_dim), dtype=dtype),
        jnp.zeros((batch, heads, seq_len, head_dim), dtype=dtype),
        jnp.zeros((batch, heads, seq_len, head_dim), dtype=dtype),
    )


def _fake_gpu_device(compute_capability="8.0"):
    device = mock.MagicMock()
    device.platform = "gpu"
    device.compute_capability = compute_capability
    return device


def _fake_tpu_device():
    device = mock.MagicMock()
    device.platform = "tpu"
    # TPUs don't expose compute_capability at all; splash's checks don't
    # rely on it, but make sure it's absent rather than a stray MagicMock
    # attribute that would look "truthy".
    del device.compute_capability
    return device


# splash requires head_dim to be a multiple of 128; use that everywhere a
# splash-compatible shape is needed so dtype/bias/local_window checks are
# exercised in isolation from the head_dim requirement.
_SPLASH_HEAD_DIM = 128


# ---------------------------------------------------------------------------
# Public API surface.
# ---------------------------------------------------------------------------

def test_flash_sdpa_exposed_on_functional():
    assert hasattr(xf, "flash_sdpa")


def test_jax_flash_attention_kernel_not_reachable_from_loom():
    # Regression guard: the "jax" backend's kernel is private to
    # xera._kernel and must never be reachable as a standalone public
    # symbol, same as the cuDNN/splash backends never were.
    assert not hasattr(xl, "jax_flash_attention")


def test_invalid_backend_raises_value_error():
    q, k, v = _make_qkv(1, 2, 8, 8)
    with pytest.raises(ValueError, match="backend must be one of"):
        xf.flash_sdpa(q, k, v, backend="bogus")


def test_naive_backend_no_longer_a_valid_option():
    # "naive" used to be a valid backend value; "jax" replaced it.
    q, k, v = _make_qkv(1, 2, 8, 8)
    with pytest.raises(ValueError, match="backend must be one of"):
        xf.flash_sdpa(q, k, v, backend="naive")


def test_none_is_no_longer_a_valid_backend_value():
    # backend used to default to/accept None for auto-dispatch; "auto" replaced it.
    q, k, v = _make_qkv(1, 2, 8, 8)
    with pytest.raises(ValueError, match="backend must be one of"):
        xf.flash_sdpa(q, k, v, backend=None)


# ---------------------------------------------------------------------------
# Dispatcher: CPU always uses the "jax" backend silently (nothing to fall back from).
# ---------------------------------------------------------------------------

def test_auto_on_cpu_uses_jax_backend_and_is_correct():
    # This test environment's real platform is CPU. "jax" is the only
    # backend there, so this should just work (no exception).
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    out = xf.flash_sdpa(q, k, v, causal=True)
    assert out.shape == q.shape


def test_auto_on_cpu_prints_nothing_even_with_verbose():
    # CPU isn't a fallback from anything -- "jax" is simply the only
    # option there, so no "XeraInfo" line should appear, verbose or not.
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        xf.flash_sdpa(q, k, v, causal=True, verbose=True)
    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Dispatcher: forcing a backend never falls back and never prints.
# ---------------------------------------------------------------------------

def test_forced_cudnn_backend_raises_without_gpu_support():
    # Forcing backend="cudnn" on CPU (no GPU at all) must raise, never
    # fall back to "jax" -- forcing means "use exactly this, or fail".
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    with pytest.raises(ValueError, match="cudnn"):
        xf.flash_sdpa(q, k, v, backend="cudnn")


def test_forced_splash_backend_raises_on_unsupported_dtype():
    q, k, v = _make_qkv(1, 2, 8, _SPLASH_HEAD_DIM, dtype=jnp.float32)
    with pytest.raises(ValueError, match="splash"):
        xf.flash_sdpa(q, k, v, backend="splash")


def test_forced_cudnn_backend_accepts_bias():
    # cuDNN's fused kernel supports additive bias natively (forwarded via
    # jax.nn.dot_product_attention); forcing backend="cudnn" with a bias
    # must not raise on an otherwise-compatible (Ampere+, bf16) call.
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 2, 8, 8), dtype=jnp.bfloat16)
    fake_device = _fake_gpu_device("8.0")
    with mock.patch("jax.devices", return_value=[fake_device]):
        with mock.patch(
            "xera._kernel.flash_attention.flash_sdpa.jax.nn.dot_product_attention",
            return_value=jnp.zeros((1, 8, 2, 8), dtype=jnp.bfloat16),
        ) as mocked:
            out = xf.flash_sdpa(q, k, v, bias=bias, backend="cudnn")
    assert out.shape == q.shape
    _, kwargs = mocked.call_args
    assert kwargs["bias"] is not None
    assert kwargs["bias"].shape == (1, 8, 2, 8)  # bias transposed to (B, T, N, S)


def test_forced_cudnn_backend_accepts_local_window():
    # Likewise, cuDNN natively supports sliding-window attention; forcing
    # backend="cudnn" with local_window_size must not raise.
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.bfloat16)
    fake_device = _fake_gpu_device("8.0")
    with mock.patch("jax.devices", return_value=[fake_device]):
        with mock.patch(
            "xera._kernel.flash_attention.flash_sdpa.jax.nn.dot_product_attention",
            return_value=jnp.zeros((1, 8, 2, 8), dtype=jnp.bfloat16),
        ) as mocked:
            out = xf.flash_sdpa(q, k, v, local_window_size=4, backend="cudnn")
    assert out.shape == q.shape
    _, kwargs = mocked.call_args
    assert kwargs["local_window_size"] == 4


def test_forced_cudnn_backend_rejects_old_gpu():
    # Forcing backend="cudnn" on a pre-Ampere GPU must raise, even with a
    # perfectly fine dtype -- sm_80+ is a hard cuDNN requirement.
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.bfloat16)
    fake_device = _fake_gpu_device("7.5")
    with mock.patch("jax.devices", return_value=[fake_device]):
        with pytest.raises(ValueError, match="sm_80"):
            xf.flash_sdpa(q, k, v, backend="cudnn")


def test_forced_splash_backend_rejects_bias():
    q, k, v = _make_qkv(1, 2, 8, _SPLASH_HEAD_DIM, dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 2, 8, 8), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="bias"):
        xf.flash_sdpa(q, k, v, bias=bias, backend="splash")


def test_forced_splash_backend_rejects_local_window():
    q, k, v = _make_qkv(1, 2, 8, _SPLASH_HEAD_DIM, dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="local_window_size"):
        xf.flash_sdpa(q, k, v, local_window_size=4, backend="splash")


def test_forced_splash_backend_rejects_non_128_multiple_head_dim():
    q, k, v = _make_qkv(1, 2, 8, 96, dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="128"):
        xf.flash_sdpa(q, k, v, backend="splash")


def test_forced_backend_never_prints_even_when_raising_and_verbose():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pytest.raises(ValueError):
            xf.flash_sdpa(q, k, v, backend="cudnn", verbose=True)
    assert buf.getvalue() == ""


def test_forced_jax_backend_works_directly():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    out = xf.flash_sdpa(q, k, v, causal=True, backend="jax")
    assert out.shape == q.shape


def test_jax_backend_accepts_custom_tile_sizes():
    q, k, v = _make_qkv(1, 2, 37, 8, dtype=jnp.float32)
    out = xf.flash_sdpa(q, k, v, causal=True, backend="jax", block_q=16, block_k=16)
    assert out.shape == q.shape


def test_auto_falling_back_to_jax_accepts_custom_tile_sizes():
    q, k, v = _make_qkv(1, 2, 37, 8, dtype=jnp.float32)
    fake_device = _fake_gpu_device("7.5")  # pre-Ampere: forces fallback to "jax"
    with mock.patch("jax.devices", return_value=[fake_device]):
        out = xf.flash_sdpa(q, k, v, causal=True, block_q=16, block_k=16)
    assert out.shape == q.shape


def test_cudnn_backend_rejects_custom_tile_sizes():
    # Tile sizes are meaningless for cuDNN/splash, which manage their own
    # tiling -- silently ignoring them would be more confusing than
    # raising, so this must raise rather than a no-op.
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    with pytest.raises(ValueError, match="block_q/block_k"):
        xf.flash_sdpa(q, k, v, backend="cudnn", block_q=16, block_k=16)


def test_splash_backend_rejects_custom_tile_sizes():
    q, k, v = _make_qkv(1, 2, 8, _SPLASH_HEAD_DIM, dtype=jnp.float32)
    with pytest.raises(ValueError, match="block_q/block_k"):
        xf.flash_sdpa(q, k, v, backend="splash", block_q=16, block_k=16)


def test_forced_jax_backend_prints_nothing_even_with_verbose():
    q, k, v = _make_qkv(1, 2, 8, 8, dtype=jnp.float32)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        xf.flash_sdpa(q, k, v, causal=True, backend="jax", verbose=True)
    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Dispatcher: verbose=False (the default) is silent no matter what happens.
# ---------------------------------------------------------------------------

def test_auto_on_simulated_gpu_fallback_prints_nothing_by_default():
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.float32)
    fake_device = _fake_gpu_device("8.0")

    buf = io.StringIO()
    with mock.patch("jax.devices", return_value=[fake_device]):
        with contextlib.redirect_stdout(buf):
            out = xf.flash_sdpa(q, k, v, causal=True)  # verbose defaults to False

    assert out.shape == q.shape
    assert buf.getvalue() == ""


def test_auto_on_simulated_tpu_fallback_prints_nothing_by_default():
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.float32)
    fake_device = _fake_tpu_device()

    buf = io.StringIO()
    with mock.patch("jax.devices", return_value=[fake_device]):
        with contextlib.redirect_stdout(buf):
            out = xf.flash_sdpa(q, k, v, causal=True)  # verbose defaults to False

    assert out.shape == q.shape
    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Dispatcher: with verbose=True, an incompatible dtype/hardware/feature
# falls back to "jax" and prints exactly one "XeraInfo" line explaining why.
# ---------------------------------------------------------------------------

def test_auto_on_simulated_gpu_with_unsupported_dtype_falls_back_and_prints():
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.float32)
    fake_device = _fake_gpu_device("8.0")

    buf = io.StringIO()
    with mock.patch("jax.devices", return_value=[fake_device]):
        with contextlib.redirect_stdout(buf):
            out = xf.flash_sdpa(q, k, v, causal=True, verbose=True)

    assert out.shape == q.shape
    printed = buf.getvalue()
    assert printed.startswith("XeraInfo:")
    assert "backend='jax'" in printed
    assert "float32" in printed
    assert len(printed.strip().splitlines()) == 1


def test_auto_on_simulated_old_gpu_falls_back_and_prints():
    # Compute capability below sm_80 should fall back even for an
    # otherwise-perfectly-fine bf16 call.
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.bfloat16)
    fake_device = _fake_gpu_device("7.5")

    buf = io.StringIO()
    with mock.patch("jax.devices", return_value=[fake_device]):
        with contextlib.redirect_stdout(buf):
            out = xf.flash_sdpa(q, k, v, causal=True, verbose=True)

    assert out.shape == q.shape
    printed = buf.getvalue()
    assert printed.startswith("XeraInfo:")
    assert "sm_80" in printed
    assert "sm_75" in printed


def test_auto_on_simulated_gpu_with_undetectable_compute_capability_falls_back_and_prints():
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.bfloat16)
    fake_device = mock.MagicMock()
    fake_device.platform = "gpu"
    del fake_device.compute_capability  # simulate an old jaxlib without this attribute

    buf = io.StringIO()
    with mock.patch("jax.devices", return_value=[fake_device]):
        with contextlib.redirect_stdout(buf):
            out = xf.flash_sdpa(q, k, v, causal=True, verbose=True)

    assert out.shape == q.shape
    printed = buf.getvalue()
    assert printed.startswith("XeraInfo:")
    assert "could not be determined" in printed


def test_auto_on_simulated_gpu_with_bias_uses_cudnn_and_prints_nothing():
    # cuDNN supports bias natively, so backend="auto" on a compatible GPU
    # should route straight to cuDNN -- no fallback, nothing printed, even
    # with verbose=True (a successful vendor-backend dispatch never prints).
    q, k, v = _make_qkv(1, 2, 16, 8, dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 2, 16, 16), dtype=jnp.bfloat16)
    fake_device = _fake_gpu_device("8.0")

    buf = io.StringIO()
    with mock.patch("jax.devices", return_value=[fake_device]):
        with mock.patch(
            "xera._kernel.flash_attention.flash_sdpa.jax.nn.dot_product_attention",
            return_value=jnp.zeros((1, 16, 2, 8), dtype=jnp.bfloat16),
        ) as mocked:
            with contextlib.redirect_stdout(buf):
                out = xf.flash_sdpa(q, k, v, bias=bias, verbose=True)

    assert out.shape == q.shape
    assert buf.getvalue() == ""
    _, kwargs = mocked.call_args
    assert kwargs["bias"] is not None


def test_auto_on_simulated_tpu_with_unsupported_dtype_falls_back_and_prints():
    q, k, v = _make_qkv(1, 2, 16, _SPLASH_HEAD_DIM, dtype=jnp.float32)
    fake_device = _fake_tpu_device()

    buf = io.StringIO()
    with mock.patch("jax.devices", return_value=[fake_device]):
        with contextlib.redirect_stdout(buf):
            out = xf.flash_sdpa(q, k, v, causal=True, verbose=True)

    assert out.shape == q.shape
    printed = buf.getvalue()
    assert printed.startswith("XeraInfo:")
    assert "float32" in printed


def test_auto_on_simulated_tpu_with_non_128_multiple_head_dim_falls_back_and_prints():
    q, k, v = _make_qkv(1, 2, 16, 96, dtype=jnp.bfloat16)
    fake_device = _fake_tpu_device()

    buf = io.StringIO()
    with mock.patch("jax.devices", return_value=[fake_device]):
        with contextlib.redirect_stdout(buf):
            out = xf.flash_sdpa(q, k, v, causal=True, verbose=True)

    assert out.shape == q.shape
    printed = buf.getvalue()
    assert printed.startswith("XeraInfo:")
    assert "128" in printed


def test_auto_on_simulated_tpu_with_local_window_falls_back_and_prints():
    q, k, v = _make_qkv(1, 2, 16, _SPLASH_HEAD_DIM, dtype=jnp.bfloat16)
    fake_device = _fake_tpu_device()

    buf = io.StringIO()
    with mock.patch("jax.devices", return_value=[fake_device]):
        with contextlib.redirect_stdout(buf):
            out = xf.flash_sdpa(q, k, v, local_window_size=4, verbose=True)

    assert out.shape == q.shape
    printed = buf.getvalue()
    assert printed.startswith("XeraInfo:")
    assert "local_window_size" in printed


# ---------------------------------------------------------------------------
# _parse_compute_capability: best-effort parsing of a GPU device's
# `compute_capability` attribute.
# ---------------------------------------------------------------------------

def test_parse_compute_capability_parses_major_minor_string():
    device = mock.MagicMock()
    device.compute_capability = "8.6"
    assert _parse_compute_capability(device) == (8, 6)


def test_parse_compute_capability_returns_none_when_attribute_absent():
    device = mock.MagicMock()
    del device.compute_capability
    assert _parse_compute_capability(device) is None


def test_parse_compute_capability_returns_none_on_unparseable_value():
    device = mock.MagicMock()
    device.compute_capability = "not-a-version"
    assert _parse_compute_capability(device) is None


# ---------------------------------------------------------------------------
# cuDNN / splash preflight compatibility checks (unit-tested directly,
# since this environment has no GPU/TPU to exercise the real backends).
# ---------------------------------------------------------------------------

def test_cudnn_rejects_fp32():
    q_fp32 = jnp.zeros((1, 2, 8, 4), dtype=jnp.float32)
    device = _fake_gpu_device("8.0")
    issue = _cudnn_compatibility_issue(q_fp32, device=device, bias=None, local_window_size=None)
    assert issue is not None
    assert "float32" in issue
    assert "cuDNN" in issue


def test_cudnn_accepts_bf16_on_ampere_with_no_extra_features():
    q_bf16 = jnp.zeros((1, 2, 8, 4), dtype=jnp.bfloat16)
    device = _fake_gpu_device("8.0")
    issue = _cudnn_compatibility_issue(q_bf16, device=device, bias=None, local_window_size=None)
    assert issue is None


def test_cudnn_accepts_bf16_on_newer_than_ampere():
    q_bf16 = jnp.zeros((1, 2, 8, 4), dtype=jnp.bfloat16)
    device = _fake_gpu_device("9.0")  # Hopper
    issue = _cudnn_compatibility_issue(q_bf16, device=device, bias=None, local_window_size=None)
    assert issue is None


def test_cudnn_rejects_pre_ampere_gpu():
    q_bf16 = jnp.zeros((1, 2, 8, 4), dtype=jnp.bfloat16)
    device = _fake_gpu_device("7.5")  # Turing
    issue = _cudnn_compatibility_issue(q_bf16, device=device, bias=None, local_window_size=None)
    assert issue is not None
    assert "sm_75" in issue
    assert "sm_80" in issue


def test_cudnn_rejects_when_compute_capability_undetectable():
    q_bf16 = jnp.zeros((1, 2, 8, 4), dtype=jnp.bfloat16)
    device = mock.MagicMock()
    del device.compute_capability
    issue = _cudnn_compatibility_issue(q_bf16, device=device, bias=None, local_window_size=None)
    assert issue is not None
    assert "could not be determined" in issue


def test_cudnn_accepts_bias():
    # cuDNN's fused kernel supports additive bias natively; the preflight
    # check must not reject it.
    q_bf16 = jnp.zeros((1, 2, 8, 4), dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 2, 8, 8))
    device = _fake_gpu_device("8.0")
    issue = _cudnn_compatibility_issue(q_bf16, device=device, bias=bias, local_window_size=None)
    assert issue is None


def test_cudnn_accepts_local_window():
    # Likewise for local_window_size -- cuDNN lowers it to native sliding
    # window attention.
    q_bf16 = jnp.zeros((1, 2, 8, 4), dtype=jnp.bfloat16)
    device = _fake_gpu_device("8.0")
    issue = _cudnn_compatibility_issue(q_bf16, device=device, bias=None, local_window_size=4)
    assert issue is None


def test_splash_accepts_plain_request():
    q_bf16 = jnp.zeros((1, 2, 8, _SPLASH_HEAD_DIM), dtype=jnp.bfloat16)
    issue = _splash_compatibility_issue(q_bf16, bias=None, local_window_size=None)
    assert issue is None


def test_splash_rejects_fp32():
    q_fp32 = jnp.zeros((1, 2, 8, _SPLASH_HEAD_DIM), dtype=jnp.float32)
    issue = _splash_compatibility_issue(q_fp32, bias=None, local_window_size=None)
    assert issue is not None
    assert "float32" in issue


def test_splash_rejects_non_128_multiple_head_dim():
    q_bf16 = jnp.zeros((1, 2, 8, 96), dtype=jnp.bfloat16)
    issue = _splash_compatibility_issue(q_bf16, bias=None, local_window_size=None)
    assert issue is not None
    assert "128" in issue
    assert "96" in issue


def test_splash_accepts_larger_128_multiple_head_dim():
    q_bf16 = jnp.zeros((1, 2, 8, 256), dtype=jnp.bfloat16)
    issue = _splash_compatibility_issue(q_bf16, bias=None, local_window_size=None)
    assert issue is None


def test_splash_rejects_bias():
    q_bf16 = jnp.zeros((1, 2, 8, _SPLASH_HEAD_DIM), dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 2, 8, 8))
    issue = _splash_compatibility_issue(q_bf16, bias=bias, local_window_size=None)
    assert issue is not None
    assert "bias" in issue


def test_splash_rejects_local_window():
    q_bf16 = jnp.zeros((1, 2, 8, _SPLASH_HEAD_DIM), dtype=jnp.bfloat16)
    issue = _splash_compatibility_issue(q_bf16, bias=None, local_window_size=4)
    assert issue is not None
    assert "local_window_size" in issue
