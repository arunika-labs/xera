"""Tests for xera.loom.auto_flash_attention: AutoFA backend dispatch and
the naive Pallas flash-attention kernel (causal masking, additive bias,
local windowing, XeraWarning transparency)."""

import unittest.mock as mock
import warnings

import jax
import jax.numpy as jnp
import pytest
import xera.loom as loom
from xera.loom.auto_flash_attention import (
    XeraWarning,
    _cudnn_compatibility_issue,
    _flash_attention_naive,
    _splash_compatibility_issue,
)


def reference_attention(q, k, v, *, causal=False, scale=None, bias=None, window_left=None, window_right=None):
    """Plain (non-tiled) attention used as the ground truth for correctness checks."""
    head_dim = q.shape[-1]
    seq_len = q.shape[2]
    scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)
    scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
    if bias is not None:
        scores = scores + bias
    if causal:
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(mask, scores, -jnp.inf)
    if window_left is not None or window_right is not None:
        q_pos = jnp.arange(seq_len)[:, None]
        k_pos = jnp.arange(seq_len)[None, :]
        rel = q_pos - k_pos
        window_mask = jnp.ones((seq_len, seq_len), dtype=bool)
        if window_left is not None:
            window_mask &= rel <= window_left
        if window_right is not None:
            window_mask &= -rel <= window_right
        scores = jnp.where(window_mask, scores, -jnp.inf)
    attn = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhts,bhsd->bhtd", attn, v)


@pytest.fixture(autouse=True)
def _silence_warnings_by_default():
    # Individual tests that care about warnings opt in via
    # warnings.catch_warnings themselves; this just keeps pytest output
    # clean for the tests that don't.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XeraWarning)
        yield


# ---------------------------------------------------------------------------
# Naive kernel: correctness against a plain reference implementation.
# ---------------------------------------------------------------------------

def _make_qkv(key, batch, heads, seq_len, head_dim):
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (batch, heads, seq_len, head_dim))
    k = jax.random.normal(kk, (batch, heads, seq_len, head_dim))
    v = jax.random.normal(kv, (batch, heads, seq_len, head_dim))
    return q, k, v


def test_naive_matches_reference_plain():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 4, 37, 16)
    out = _flash_attention_naive(q, k, v, block_q=16, block_k=16, interpret=True)
    ref = reference_attention(q, k, v)
    assert jnp.allclose(out, ref, atol=1e-4)


def test_naive_matches_reference_causal():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 4, 37, 16)
    out = _flash_attention_naive(q, k, v, causal=True, block_q=16, block_k=16, interpret=True)
    ref = reference_attention(q, k, v, causal=True)
    assert jnp.allclose(out, ref, atol=1e-4)


def test_naive_matches_reference_with_bias():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 3, 37, 16)
    bias = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 37, 37)) * 0.1
    out = _flash_attention_naive(q, k, v, bias=bias, block_q=16, block_k=16, interpret=True)
    ref = reference_attention(q, k, v, bias=bias)
    assert jnp.allclose(out, ref, atol=1e-4)


def test_naive_matches_reference_bias_and_causal():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 3, 37, 16)
    bias = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 37, 37)) * 0.1
    out = _flash_attention_naive(q, k, v, bias=bias, causal=True, block_q=16, block_k=16, interpret=True)
    ref = reference_attention(q, k, v, bias=bias, causal=True)
    assert jnp.allclose(out, ref, atol=1e-4)


@pytest.mark.parametrize(
    "window_left,window_right",
    [(5, 5), (5, None), (None, 5), (0, 0)],
)
def test_naive_matches_reference_local_window(window_left, window_right):
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 3, 37, 16)
    window = (window_left, window_right)
    out = _flash_attention_naive(
        q, k, v, local_window_size=window, block_q=16, block_k=16, interpret=True
    )
    ref = reference_attention(q, k, v, window_left=window_left, window_right=window_right)
    assert jnp.allclose(out, ref, atol=1e-4)


def test_naive_matches_reference_local_window_symmetric_int():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 3, 37, 16)
    out = _flash_attention_naive(q, k, v, local_window_size=5, block_q=16, block_k=16, interpret=True)
    ref = reference_attention(q, k, v, window_left=5, window_right=5)
    assert jnp.allclose(out, ref, atol=1e-4)


def test_naive_matches_reference_local_window_and_causal():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 2, 3, 37, 16)
    out = _flash_attention_naive(
        q, k, v, local_window_size=5, causal=True, block_q=16, block_k=16, interpret=True
    )
    ref = reference_attention(q, k, v, causal=True, window_left=5, window_right=5)
    assert jnp.allclose(out, ref, atol=1e-4)


def test_naive_no_nans_with_narrow_window():
    # Regression test: a k-block that is entirely masked out (fully outside
    # the local window) used to produce exp(-inf - (-inf)) = NaN via the
    # online-softmax running-max update.
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 1, 1, 37, 4)
    out = _flash_attention_naive(
        q, k, v, local_window_size=5, block_q=16, block_k=16, interpret=True
    )
    assert not jnp.any(jnp.isnan(out))


@pytest.mark.parametrize(
    "batch,heads,seq_len,head_dim,block_q,block_k",
    [
        (2, 4, 37, 16, 16, 16),   # seq_len not divisible by either block size
        (1, 1, 8, 4, 8, 8),       # single block
        (2, 4, 64, 16, 16, 16),   # evenly divisible
        (1, 2, 100, 8, 32, 17),   # mismatched, irregular block sizes
        (1, 1, 1, 4, 8, 8),       # seq_len smaller than block size
        (1, 1, 5, 4, 8, 8),       # seq_len smaller than block size
    ],
)
def test_naive_matches_reference_various_shapes(batch, heads, seq_len, head_dim, block_q, block_k):
    q, k, v = _make_qkv(jax.random.PRNGKey(0), batch, heads, seq_len, head_dim)
    for causal in (False, True):
        out = _flash_attention_naive(
            q, k, v, causal=causal, block_q=block_q, block_k=block_k, interpret=True
        )
        ref = reference_attention(q, k, v, causal=causal)
        assert jnp.allclose(out, ref, atol=1e-4), f"mismatch: causal={causal}"


def test_naive_output_shape_matches_input():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 3, 2, 29, 8)
    out = _flash_attention_naive(q, k, v, block_q=16, block_k=16, interpret=True)
    assert out.shape == q.shape


# ---------------------------------------------------------------------------
# Dispatcher: backend selection and forcing.
# ---------------------------------------------------------------------------

def test_auto_flash_attention_exposed_on_loom():
    assert hasattr(loom, "auto_flash_attention")
    assert hasattr(loom, "XeraWarning")


def test_auto_dispatches_to_naive_on_cpu_and_is_correct():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 1, 2, 16, 8)
    out = loom.auto_flash_attention(q, k, v, causal=True)
    ref = reference_attention(q, k, v, causal=True)
    assert jnp.allclose(out, ref, atol=1e-4)


def test_explicit_naive_backend_matches_auto_on_cpu():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 1, 2, 16, 8)
    out_auto = loom.auto_flash_attention(q, k, v, causal=True)
    out_explicit = loom.auto_flash_attention(q, k, v, causal=True, backend="naive")
    assert jnp.allclose(out_auto, out_explicit, atol=1e-4)


def test_explicit_naive_backend_supports_bias_and_window():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 1, 2, 20, 8)
    bias = jax.random.normal(jax.random.PRNGKey(1), (1, 2, 20, 20)) * 0.1
    out = loom.auto_flash_attention(q, k, v, bias=bias, local_window_size=4, backend="naive")
    ref = reference_attention(q, k, v, bias=bias, window_left=4, window_right=4)
    assert jnp.allclose(out, ref, atol=1e-4)


def test_invalid_backend_raises_value_error():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 1, 2, 8, 8)
    with pytest.raises(ValueError, match="backend must be one of"):
        loom.auto_flash_attention(q, k, v, backend="bogus")


def test_forced_cudnn_backend_raises_without_gpu():
    # No real GPU in the test environment -- forcing backend="cudnn" should
    # surface an error from jax.nn.dot_product_attention rather than
    # silently falling back (forcing a backend means "use exactly this,
    # or fail").
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 1, 2, 8, 8)
    with pytest.raises(Exception):
        loom.auto_flash_attention(q, k, v, backend="cudnn")


# ---------------------------------------------------------------------------
# XeraWarning: transparency around backend selection and fallback.
# ---------------------------------------------------------------------------

def test_auto_on_cpu_emits_backend_selected_warning():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 1, 2, 8, 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loom.auto_flash_attention(q, k, v)
    messages = [str(w.message) for w in caught]
    assert any("using 'naive' backend" in m for m in messages)
    assert all(w.category is XeraWarning for w in caught)


def test_auto_on_cpu_emits_interpret_mode_warning():
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 1, 2, 8, 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loom.auto_flash_attention(q, k, v)
    messages = [str(w.message) for w in caught]
    assert any("interpret mode" in m for m in messages)


def test_env_var_silences_all_autofa_warnings(monkeypatch):
    monkeypatch.setenv("XERA_SILENCE_AUTOFA_WARNINGS", "1")
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 1, 2, 8, 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loom.auto_flash_attention(q, k, v)
    assert len([w for w in caught if w.category is XeraWarning]) == 0


def test_explicit_backend_naive_still_warns_about_interpret_mode():
    # Forcing backend="naive" bypasses the auto-selection warning, but the
    # interpret-mode warning is orthogonal (about *how* naive is running,
    # not *that* it was chosen) and should still fire.
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 1, 2, 8, 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loom.auto_flash_attention(q, k, v, backend="naive")
    messages = [str(w.message) for w in caught]
    assert any("interpret mode" in m for m in messages)


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


def test_auto_fp32_on_simulated_gpu_falls_back_with_dtype_warning(monkeypatch):
    # Simulate a GPU platform (mocking jax.devices()) to exercise the
    # fp32-on-GPU fallback path without needing real GPU hardware. Also
    # force interpret=True on the naive kernel call, since this test
    # environment's *actual* backend is CPU (which only runs Pallas in
    # interpret mode) even though we're pretending the platform is "gpu"
    # for dispatch purposes.
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 1, 2, 16, 8)

    fake_device = mock.MagicMock()
    fake_device.platform = "gpu"

    import sys
    afa_module = sys.modules["xera.loom.auto_flash_attention"]
    real_naive = afa_module._flash_attention_naive

    def forced_interpret_naive(*args, **kwargs):
        kwargs["interpret"] = True
        return real_naive(*args, **kwargs)

    monkeypatch.setattr(afa_module, "_flash_attention_naive", forced_interpret_naive)

    with mock.patch("jax.devices", return_value=[fake_device]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = loom.auto_flash_attention(q, k, v, causal=True)

    messages = [str(w.message) for w in caught]
    assert any("float32" in m and "naive" in m for m in messages)
    ref = reference_attention(q, k, v, causal=True)
    assert jnp.allclose(out, ref, atol=1e-4)


def test_auto_bias_on_simulated_gpu_falls_back_with_feature_warning(monkeypatch):
    q, k, v = _make_qkv(jax.random.PRNGKey(0), 1, 2, 16, 8)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    bias = jax.random.normal(jax.random.PRNGKey(1), (1, 2, 16, 16)).astype(jnp.bfloat16) * 0.1

    fake_device = mock.MagicMock()
    fake_device.platform = "gpu"

    import sys
    afa_module = sys.modules["xera.loom.auto_flash_attention"]
    real_naive = afa_module._flash_attention_naive

    def forced_interpret_naive(*args, **kwargs):
        kwargs["interpret"] = True
        return real_naive(*args, **kwargs)

    monkeypatch.setattr(afa_module, "_flash_attention_naive", forced_interpret_naive)

    with mock.patch("jax.devices", return_value=[fake_device]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loom.auto_flash_attention(q, k, v, bias=bias)

    messages = [str(w.message) for w in caught]
    assert any("bias" in m and "naive" in m for m in messages)
