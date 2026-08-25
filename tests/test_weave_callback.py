"""Tests for xera.weave.Callback.print (every-N throttling via
jax.lax.cond) and xera.weave.Callback.log (buffered, batched flush to
a .jsonl file, also via jax.lax.cond)."""

import glob
import json
import os
import jax
import jax.numpy as jnp
import pytest
from xera.weave.callback import Callback
from xera.weave.loop import loop


def _read_jsonl(path, name="log"):
    p = os.path.join(path, f"{name}.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Callback.print
# ---------------------------------------------------------------------------

def test_print_every_one_prints_every_step(capsys):
    def body(carry, i):
        Callback.print(i, "x={x}", x=i)
        return carry, i

    jax.lax.scan(body, 0, jnp.arange(4))
    jax.effects_barrier()
    out = capsys.readouterr().out
    for step in range(4):
        assert f"step={step}" in out


def test_print_every_n_throttles_via_lax_cond(capsys):
    def body(carry, i):
        Callback.print(i, "x={x}", every=3, x=i)
        return carry, i

    jax.lax.scan(body, 0, jnp.arange(7))
    jax.effects_barrier()
    out = capsys.readouterr().out
    for step in (0, 3, 6):
        assert f"step={step}" in out
    for step in (1, 2, 4, 5):
        assert f"step={step} " not in out


def test_print_default_fmt_prints_all_values(capsys):
    def body(carry, i):
        Callback.print(i, loss=i, lr=1)
        return carry, i

    jax.lax.scan(body, 0, jnp.arange(2))
    jax.effects_barrier()
    out = capsys.readouterr().out
    assert "loss=" in out
    assert "lr=" in out


def test_print_works_inside_fori_loop(capsys):
    def body(i, carry):
        Callback.print(i, "v={v}", every=2, v=i)
        return carry

    jax.lax.fori_loop(0, 4, body, 0)
    jax.effects_barrier()
    out = capsys.readouterr().out
    assert "step=0" in out
    assert "step=2" in out


# ---------------------------------------------------------------------------
# Callback.log -- buffer shape / init
# ---------------------------------------------------------------------------

def test_log_init_buffer_has_expected_shape(tmp_path):
    log_fn, buf0 = Callback.log(str(tmp_path), every=10, loss=jnp.float32, lr=jnp.float32)
    assert buf0["step"].shape == (10,)
    assert buf0["values"]["loss"].shape == (10,)
    assert buf0["values"]["lr"].shape == (10,)
    assert int(buf0["count"]) == 0


def test_log_rejects_missing_or_extra_fields(tmp_path):
    log_fn, buf0 = Callback.log(str(tmp_path), every=5, loss=jnp.float32)
    with pytest.raises(ValueError):
        log_fn(buf0, 0, wrong_field=1.0)
    with pytest.raises(ValueError):
        log_fn(buf0, 0, loss=1.0, extra=2.0)


# ---------------------------------------------------------------------------
# Callback.log -- flush behavior
# ---------------------------------------------------------------------------

def test_log_flushes_only_once_buffer_is_full(tmp_path):
    path = str(tmp_path)
    log_fn, buf0 = Callback.log(path, every=3, loss=jnp.float32)

    def body(buf, i):
        loss = i.astype(jnp.float32) * 0.1
        buf = log_fn(buf, i, loss=loss)
        return buf, i

    final_buf, _ = jax.lax.scan(body, buf0, jnp.arange(7))
    jax.effects_barrier()

    records = _read_jsonl(path)
    # 7 steps, every=3 -> two full flushes (steps 0-2, 3-5), step 6 still pending
    assert len(records) == 6
    assert [r["step"] for r in records] == [0, 1, 2, 3, 4, 5]
    assert int(final_buf["count"]) == 1  # step 6 sitting in the buffer, unflushed


def test_log_records_correct_values(tmp_path):
    path = str(tmp_path)
    log_fn, buf0 = Callback.log(path, every=4, loss=jnp.float32, lr=jnp.float32)

    def body(buf, i):
        loss = i.astype(jnp.float32) * 2.0
        lr = jnp.float32(0.01)
        buf = log_fn(buf, i, loss=loss, lr=lr)
        return buf, i

    jax.lax.scan(body, buf0, jnp.arange(4))
    jax.effects_barrier()

    records = _read_jsonl(path)
    assert len(records) == 4
    for i, r in enumerate(records):
        assert r["step"] == i
        assert abs(r["loss"] - i * 2.0) < 1e-5
        assert abs(r["lr"] - 0.01) < 1e-5


def test_log_appends_across_multiple_full_batches(tmp_path):
    path = str(tmp_path)
    log_fn, buf0 = Callback.log(path, every=2, loss=jnp.float32)

    def body(buf, i):
        buf = log_fn(buf, i, loss=i.astype(jnp.float32))
        return buf, i

    jax.lax.scan(body, buf0, jnp.arange(10))
    jax.effects_barrier()

    records = _read_jsonl(path)
    assert len(records) == 10
    assert [r["step"] for r in records] == list(range(10))


def test_log_custom_name_writes_separate_file(tmp_path):
    path = str(tmp_path)
    log_fn, buf0 = Callback.log(path, every=2, name="metrics", loss=jnp.float32)

    def body(buf, i):
        buf = log_fn(buf, i, loss=i.astype(jnp.float32))
        return buf, i

    jax.lax.scan(body, buf0, jnp.arange(4))
    jax.effects_barrier()

    assert os.path.exists(os.path.join(path, "metrics.jsonl"))
    assert not os.path.exists(os.path.join(path, "log.jsonl"))


def test_log_works_inside_fori_loop(tmp_path):
    path = str(tmp_path)
    log_fn, buf0 = Callback.log(path, every=2, loss=jnp.float32)

    def body(i, buf):
        return log_fn(buf, i, loss=i.astype(jnp.float32))

    jax.lax.fori_loop(0, 5, body, buf0)
    jax.effects_barrier()

    records = _read_jsonl(path)
    assert len(records) == 4  # steps 0-3 flushed in two batches, step 4 pending


# ---------------------------------------------------------------------------
# End-to-end: Callback.log's buffer threaded through weave.loop's carry,
# alongside a real training step.
# ---------------------------------------------------------------------------

def test_log_threaded_through_weave_loop_carry(tmp_path):
    path = str(tmp_path)
    log_fn, buf0 = Callback.log(path, every=5, loss=jnp.float32)

    def step_fn(carry, i):
        counter, buf = carry
        loss = counter.astype(jnp.float32)
        buf = log_fn(buf, i, loss=loss)
        return (counter + 1, buf), loss

    (final_counter, final_buf), losses = loop(
        step_fn, (jnp.asarray(0), buf0), type="scan", steps=12,
    )
    jax.effects_barrier()

    records = _read_jsonl(path)
    assert len(records) == 10  # two full flushes of 5 within 12 steps
    assert int(final_counter) == 12
