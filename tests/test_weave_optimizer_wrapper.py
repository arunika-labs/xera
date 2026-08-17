"""Tests for xera.weave.optimizer.wrapper: Clip, Schedule, Accumulate,
WeightDecay, EMA, Freeze, Lookahead, Cast."""

import jax
import jax.numpy as jnp
import pytest
import xera.weave as weave
from xera.weave.optimizer.core.sgd import SGDMomentum
from xera.weave.optimizer.core.adam import Adam, AdamW
from xera.weave.optimizer.base import apply_updates
from xera.weave.optimizer.partition import Partition
from xera.weave.optimizer.wrapper.clip import Clip
from xera.weave.optimizer.wrapper.schedule import Schedule
from xera.weave.optimizer.wrapper.accumulate import Accumulate
from xera.weave.optimizer.wrapper.weight_decay import WeightDecay
from xera.weave.optimizer.wrapper.ema import EMA
from xera.weave.optimizer.wrapper.freeze import Freeze
from xera.weave.optimizer.wrapper.lookahead import Lookahead
from xera.weave.optimizer.wrapper.cast import Cast


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------

def test_clip_rescales_grads_above_threshold():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = Clip(threshold=1.0)(inner)
    params = {"w": jnp.zeros((2,))}
    state = opt.init(params)
    grads = {"w": jnp.array([3.0, 4.0])}  # norm = 5.0
    updates, _ = opt.update(grads, state, params)
    # Direction should be rescaled to unit norm (threshold=1.0), so the
    # SGD(lr=1.0, momentum=0) update equals -clipped_grad.
    update_norm = jnp.sqrt(jnp.sum(updates["w"] ** 2))
    assert jnp.allclose(update_norm, 1.0, atol=1e-4)


def test_clip_leaves_small_grads_unchanged():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = Clip(threshold=10.0)(inner)
    params = {"w": jnp.zeros((2,))}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0, 0.0])}  # norm well below threshold
    updates, _ = opt.update(grads, state, params)
    assert jnp.allclose(updates["w"], -grads["w"], atol=1e-4)


def test_clip_delegates_init_to_inner():
    inner = Adam(lr=0.1)
    opt = Clip(threshold=1.0)(inner)
    params = {"w": jnp.ones((3,))}
    state = opt.init(params)
    inner_state = inner.init(params)
    assert type(state) is type(inner_state)


def test_clip_composes_with_other_optimizers():
    opt = Clip(threshold=1.0)(Adam(lr=0.1))
    params = {"w": jnp.ones((3, 3)) * 5.0}
    state = opt.init(params)
    for _ in range(5):
        grads = jax.tree_util.tree_map(lambda p: 2 * p, params)
        updates, state = opt.update(grads, state, params)
        params = apply_updates(params, updates)
    assert jnp.all(jnp.isfinite(params["w"]))


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def test_schedule_scales_updates_by_fn_of_step():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = Schedule(lambda step: 0.5)(inner)
    params = {"w": jnp.zeros((2,))}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0, 1.0])}
    updates, _ = opt.update(grads, state, params)
    assert jnp.allclose(updates["w"], -0.5 * grads["w"])


def test_schedule_uses_internal_step_when_none_passed():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    seen_steps = []

    def fn(step):
        seen_steps.append(int(step))
        return 1.0

    opt = Schedule(fn)(inner)
    params = {"w": jnp.zeros((2,))}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0, 1.0])}
    _, state = opt.update(grads, state, params)
    _, state = opt.update(grads, state, params)
    assert seen_steps == [0, 1]


def test_schedule_uses_explicit_step_argument():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = Schedule(lambda step: jnp.where(step < 5, 1.0, 0.1))(inner)
    params = {"w": jnp.zeros((2,))}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0, 1.0])}

    updates_early, _ = opt.update(grads, state, params, step=0)
    updates_late, _ = opt.update(grads, state, params, step=10)
    assert jnp.allclose(updates_early["w"], -1.0 * grads["w"])
    assert jnp.allclose(updates_late["w"], -0.1 * grads["w"])


def test_schedule_linear_warmup_example():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    warmup_steps = 10
    opt = Schedule(lambda step: jnp.minimum(1.0, (step + 1) / warmup_steps))(inner)
    params = {"w": jnp.zeros((1,))}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0])}

    updates_step0, _ = opt.update(grads, state, params, step=0)
    updates_step20, _ = opt.update(grads, state, params, step=20)
    assert abs(float(updates_step0["w"][0])) < abs(float(updates_step20["w"][0]))


# ---------------------------------------------------------------------------
# Accumulate
# ---------------------------------------------------------------------------

def test_accumulate_no_op_before_window_elapses():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = Accumulate(steps=4)(inner)
    params = {"w": jnp.zeros((2,))}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0, 1.0])}

    updates, state = opt.update(grads, state, params)
    assert jnp.allclose(updates["w"], jnp.zeros(2))
    updates, state = opt.update(grads, state, params)
    assert jnp.allclose(updates["w"], jnp.zeros(2))
    updates, state = opt.update(grads, state, params)
    assert jnp.allclose(updates["w"], jnp.zeros(2))


def test_accumulate_applies_averaged_grad_at_window_end():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = Accumulate(steps=4)(inner)
    params = {"w": jnp.zeros((2,))}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0, 1.0])}

    for _ in range(3):
        _, state = opt.update(grads, state, params)
    updates, state = opt.update(grads, state, params)
    # Average of 4 identical grads == same grad; SGD(lr=1, momentum=0)
    # applies -lr * avg_grad.
    assert jnp.allclose(updates["w"], -grads["w"], atol=1e-5)


def test_accumulate_buffer_resets_after_apply():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = Accumulate(steps=2)(inner)
    params = {"w": jnp.zeros((1,))}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0])}

    _, state = opt.update(grads, state, params)  # count=1, buf accumulates
    _, state = opt.update(grads, state, params)  # count=2, applies, resets buf
    assert jnp.allclose(state.buf["w"], jnp.zeros(1))


def test_accumulate_with_explicit_step_argument():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = Accumulate(steps=2)(inner)
    params = {"w": jnp.zeros((1,))}
    state = opt.init(params)
    grads = {"w": jnp.array([2.0])}

    updates0, state = opt.update(grads, state, params, step=0)
    assert jnp.allclose(updates0["w"], jnp.zeros(1))
    updates1, state = opt.update(grads, state, params, step=1)
    assert jnp.allclose(updates1["w"], -grads["w"], atol=1e-5)


def test_accumulate_rejects_steps_below_one():
    with pytest.raises(AssertionError):
        Accumulate(steps=0)


def test_accumulate_jit_compatible():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = Accumulate(steps=2)(inner)
    params = {"w": jnp.zeros((2,))}
    state = opt.init(params)
    grads = {"w": jnp.ones((2,))}
    step_fn = jax.jit(lambda g, s, p: opt.update(g, s, p))
    updates, state = step_fn(grads, state, params)
    updates, state = step_fn(grads, state, params)
    assert jnp.allclose(updates["w"], -grads["w"], atol=1e-5)


# ---------------------------------------------------------------------------
# WeightDecay
# ---------------------------------------------------------------------------

def test_weight_decay_infers_lr_from_inner():
    inner = Adam(lr=0.1)
    opt = WeightDecay(rate=0.1)(inner)
    assert opt.lr == pytest.approx(0.1)


def test_weight_decay_explicit_lr_overrides_inference():
    inner = Adam(lr=0.1)
    opt = WeightDecay(rate=0.1, lr=0.5)(inner)
    assert opt.lr == pytest.approx(0.5)


def test_weight_decay_raises_when_no_lr_found():
    class NoLrOptimizer:
        def init(self, params):
            return None

        def update(self, grads, state, params=None, step=None):
            return grads, state

    with pytest.raises(TypeError):
        WeightDecay(rate=0.1)(NoLrOptimizer())


def test_weight_decay_adds_extra_shrinkage_to_updates():
    inner = SGDMomentum(lr=0.1, momentum=0.0)
    opt = WeightDecay(rate=0.5)(inner)
    params = {"w": jnp.array([2.0])}
    state = opt.init(params)
    grads = {"w": jnp.array([0.0])}
    updates, _ = opt.update(grads, state, params)
    # updates = inner_updates(=0) - lr*rate*p = -0.1*0.5*2.0 = -0.1
    assert jnp.allclose(updates["w"], -0.1)


def test_weight_decay_zero_rate_is_no_op():
    inner = SGDMomentum(lr=0.1, momentum=0.0)
    opt_plain = inner
    opt_decay = WeightDecay(rate=0.0)(SGDMomentum(lr=0.1, momentum=0.0))
    params = {"w": jnp.array([2.0])}
    grads = {"w": jnp.array([0.5])}
    updates_plain, _ = opt_plain.update(grads, opt_plain.init(params), params)
    updates_decay, _ = opt_decay.update(grads, opt_decay.init(params), params)
    assert jnp.allclose(updates_plain["w"], updates_decay["w"])


def test_weight_decay_delegates_init_to_inner():
    inner = Adam(lr=0.1)
    opt = WeightDecay(rate=0.1)(inner)
    params = {"w": jnp.ones((3,))}
    state = opt.init(params)
    inner_state = inner.init(params)
    assert type(state) is type(inner_state)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def test_ema_shadow_initialized_to_params():
    inner = SGDMomentum(lr=0.1, momentum=0.0)
    opt = EMA(decay=0.9)(inner)
    params = {"w": jnp.array([1.0, 2.0])}
    state = opt.init(params)
    assert jnp.allclose(state.shadow["w"], params["w"])


def test_ema_shadow_tracks_new_params_with_decay():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = EMA(decay=0.9, warmup_steps=0)(inner)
    params = {"w": jnp.array([1.0])}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0])}
    updates, state = opt.update(grads, state, params)
    new_params = apply_updates(params, updates)
    expected_shadow = 0.9 * params["w"] + 0.1 * new_params["w"]
    assert jnp.allclose(state.shadow["w"], expected_shadow)


def test_ema_warmup_disables_decay_early():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = EMA(decay=0.9, warmup_steps=5)(inner)
    params = {"w": jnp.array([1.0])}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0])}
    # step=0 < warmup_steps=5, so decay used should be 0.0: shadow ==
    # new_params exactly (no blending with old shadow).
    updates, state = opt.update(grads, state, params, step=0)
    new_params = apply_updates(params, updates)
    assert jnp.allclose(state.shadow["w"], new_params["w"])


def test_ema_params_accessor():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = EMA(decay=0.9)(inner)
    params = {"w": jnp.array([1.0])}
    state = opt.init(params)
    assert jnp.allclose(opt.ema_params(state)["w"], params["w"])


def test_ema_requires_params_to_update_shadow():
    inner = SGDMomentum(lr=0.1, momentum=0.0)
    opt = EMA(decay=0.9)(inner)
    params = {"w": jnp.array([1.0])}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0])}
    # Without params, shadow should stay as-is (unchanged) per implementation.
    _, new_state = opt.update(grads, state, params=None)
    assert jnp.allclose(new_state.shadow["w"], state.shadow["w"])


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------

def test_freeze_zeros_updates_for_matched_predicate():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    # Freeze any leaf whose path contains 'frozen'.
    predicate = lambda path, leaf: "frozen" in jax.tree_util.keystr(path)
    opt = Freeze(predicate)(inner)
    params = {"frozen": jnp.array([1.0]), "trainable": jnp.array([1.0])}
    state = opt.init(params)
    grads = {"frozen": jnp.array([5.0]), "trainable": jnp.array([5.0])}
    updates, _ = opt.update(grads, state, params)
    assert jnp.allclose(updates["frozen"], jnp.zeros(1))
    assert not jnp.allclose(updates["trainable"], jnp.zeros(1))


def test_freeze_returns_partition_instance():
    inner = SGDMomentum(lr=0.1)
    opt = Freeze(lambda path, leaf: False)(inner)
    assert isinstance(opt, Partition)


def test_freeze_frozen_params_never_change_across_steps():
    inner = Adam(lr=0.5)
    predicate = lambda path, leaf: "frozen" in jax.tree_util.keystr(path)
    opt = Freeze(predicate)(inner)
    params = {"frozen": jnp.array([3.0]), "trainable": jnp.array([3.0])}
    state = opt.init(params)
    for _ in range(5):
        grads = {"frozen": jnp.array([1.0]), "trainable": jnp.array([1.0])}
        updates, state = opt.update(grads, state, params)
        params = apply_updates(params, updates)
    assert jnp.allclose(params["frozen"], 3.0)
    assert not jnp.allclose(params["trainable"], 3.0)


# ---------------------------------------------------------------------------
# Lookahead
# ---------------------------------------------------------------------------

def test_lookahead_requires_params_for_update():
    inner = SGDMomentum(lr=0.1)
    opt = Lookahead(k=5)(inner)
    params = {"w": jnp.ones((2,))}
    state = opt.init(params)
    grads = {"w": jnp.ones((2,))}
    with pytest.raises(ValueError):
        opt.update(grads, state, params=None)


def test_lookahead_slow_weights_initialized_to_params():
    inner = SGDMomentum(lr=0.1)
    opt = Lookahead(k=5, alpha=0.5)(inner)
    params = {"w": jnp.array([1.0, 2.0])}
    state = opt.init(params)
    assert jnp.allclose(state.slow["w"], params["w"])


def test_lookahead_syncs_at_k_steps():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = Lookahead(k=2, alpha=0.5)(inner)
    params = {"w": jnp.array([0.0])}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0])}

    # step 0: count=1, no sync (k=2)
    updates0, state = opt.update(grads, state, params)
    params0 = apply_updates(params, updates0)
    # step 1: count=2, syncs
    updates1, state = opt.update(grads, state, params0)
    params1 = apply_updates(params0, updates1)

    # After sync, fast params should equal newly-synced slow point.
    assert jnp.allclose(params1["w"], state.slow["w"], atol=1e-5)


def test_lookahead_no_sync_before_k_leaves_slow_unchanged():
    inner = SGDMomentum(lr=1.0, momentum=0.0)
    opt = Lookahead(k=5, alpha=0.5)(inner)
    params = {"w": jnp.array([0.0])}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0])}
    initial_slow = state.slow["w"]
    _, state = opt.update(grads, state, params)
    assert jnp.allclose(state.slow["w"], initial_slow)


def test_lookahead_rejects_k_below_one():
    with pytest.raises(AssertionError):
        Lookahead(k=0)


# ---------------------------------------------------------------------------
# Cast
# ---------------------------------------------------------------------------

def test_cast_grad_dtype_applied_before_inner_update():
    inner = SGDMomentum(lr=0.1, momentum=0.0)
    opt = Cast(grad_dtype=jnp.float32)(inner)
    params = {"w": jnp.array([1.0], dtype=jnp.float32)}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0], dtype=jnp.bfloat16)}
    updates, _ = opt.update(grads, state, params)
    assert updates["w"].dtype == jnp.float32


def test_cast_update_dtype_applied_after_inner_update():
    inner = SGDMomentum(lr=0.1, momentum=0.0)
    opt = Cast(update_dtype=jnp.bfloat16)(inner)
    params = {"w": jnp.array([1.0], dtype=jnp.float32)}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0], dtype=jnp.float32)}
    updates, _ = opt.update(grads, state, params)
    assert updates["w"].dtype == jnp.bfloat16


def test_cast_none_dtypes_are_no_op():
    inner = SGDMomentum(lr=0.1, momentum=0.0)
    opt = Cast()(inner)
    params = {"w": jnp.array([1.0])}
    state = opt.init(params)
    grads = {"w": jnp.array([2.0])}
    updates_cast, _ = opt.update(grads, state, params)
    updates_plain, _ = inner.update(grads, inner.init(params), params)
    assert jnp.allclose(updates_cast["w"], updates_plain["w"])
    assert updates_cast["w"].dtype == updates_plain["w"].dtype


def test_cast_delegates_init_to_inner():
    inner = Adam(lr=0.1)
    opt = Cast(grad_dtype=jnp.float32)(inner)
    params = {"w": jnp.ones((3,))}
    state = opt.init(params)
    inner_state = inner.init(params)
    assert type(state) is type(inner_state)


# ---------------------------------------------------------------------------
# Composition of multiple wrappers
# ---------------------------------------------------------------------------

def test_multiple_wrappers_compose_and_reduce_loss():
    opt = Schedule(lambda step: 1.0)(Clip(threshold=5.0)(Adam(lr=0.1)))
    params = {"w": jnp.ones((3, 3)) * 3.0}
    state = opt.init(params)
    initial_loss = float(jnp.sum(params["w"] ** 2))
    for _ in range(10):
        grads = jax.tree_util.tree_map(lambda p: 2 * p, params)
        updates, state = opt.update(grads, state, params)
        params = apply_updates(params, updates)
    final_loss = float(jnp.sum(params["w"] ** 2))
    assert final_loss < initial_loss


# ---------------------------------------------------------------------------
# Exposure from xera.weave namespace
# ---------------------------------------------------------------------------

def test_all_wrappers_exposed_on_weave():
    for name in ["Clip", "Schedule", "Accumulate", "WeightDecay", "EMA",
                 "Freeze", "Lookahead", "Cast"]:
        assert hasattr(weave, name)
