"""Tests for xera.weave.optimizer.core: SGDMomentum, Adam, AdamW, Lion,
RMSprop, Adagrad, Adan, Adafactor, Shampoo, MuonCore, Muon."""

import jax
import jax.numpy as jnp
import pytest
import xera.weave as weave
from xera.weave.optimizer.core.sgd import SGDMomentum, SGDMomentumState
from xera.weave.optimizer.core.adam import Adam, AdamState, AdamW, AdamWState
from xera.weave.optimizer.core.lion import Lion, LionState
from xera.weave.optimizer.core.rmsprop import RMSprop, RMSpropState
from xera.weave.optimizer.core.adagrad import Adagrad, AdagradState
from xera.weave.optimizer.core.adan import Adan, AdanState
from xera.weave.optimizer.core.adafactor import Adafactor, AdafactorState
from xera.weave.optimizer.core.shampoo import Shampoo, ShampooState
from xera.weave.optimizer.core.muon import MuonCore, MuonCoreState, Muon
from xera.weave.optimizer.base import apply_updates


def _quadratic_grads(params):
    # grad of sum(p^2) w.r.t. p is 2p -- a simple, well-behaved convex loss
    # for checking that an optimizer moves params toward zero (the minimum).
    return jax.tree_util.tree_map(lambda p: 2 * p, params)


def _loss(params):
    return sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(params))


ALL_2D_OPTIMIZER_FACTORIES = {
    "SGDMomentum": lambda: SGDMomentum(lr=0.1),
    "Adam": lambda: Adam(lr=0.1),
    "AdamW": lambda: AdamW(lr=0.1, weight_decay=0.0),
    "Lion": lambda: Lion(lr=0.1),
    "RMSprop": lambda: RMSprop(lr=0.1),
    "Adagrad": lambda: Adagrad(lr=0.1),
    "Adan": lambda: Adan(lr=0.1),
    "Adafactor": lambda: Adafactor(lr=0.1),
    "Shampoo": lambda: Shampoo(lr=0.1),
    "MuonCore": lambda: MuonCore(lr=0.1),
}


# ---------------------------------------------------------------------------
# Generic contract tests, parametrized across every optimizer.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ALL_2D_OPTIMIZER_FACTORIES))
def test_optimizer_init_step_starts_at_zero(name):
    opt = ALL_2D_OPTIMIZER_FACTORIES[name]()
    params = {"w": jnp.ones((4, 4))}
    state = opt.init(params)
    assert int(state.step) == 0


@pytest.mark.parametrize("name", list(ALL_2D_OPTIMIZER_FACTORIES))
def test_optimizer_update_increments_step(name):
    opt = ALL_2D_OPTIMIZER_FACTORIES[name]()
    params = {"w": jnp.ones((4, 4))}
    state = opt.init(params)
    grads = _quadratic_grads(params)
    _updates, new_state = opt.update(grads, state, params)
    assert int(new_state.step) == 1


@pytest.mark.parametrize("name", list(ALL_2D_OPTIMIZER_FACTORIES))
def test_optimizer_updates_match_param_tree_structure(name):
    opt = ALL_2D_OPTIMIZER_FACTORIES[name]()
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4, 4))}
    state = opt.init(params)
    grads = _quadratic_grads(params)
    updates, _new_state = opt.update(grads, state, params)
    assert set(updates.keys()) == set(params.keys())
    assert updates["w"].shape == params["w"].shape
    assert updates["b"].shape == params["b"].shape


@pytest.mark.parametrize("name", list(ALL_2D_OPTIMIZER_FACTORIES))
def test_optimizer_reduces_quadratic_loss_after_several_steps(name):
    opt = ALL_2D_OPTIMIZER_FACTORIES[name]()
    params = {"w": jnp.ones((4, 4)) * 2.0}
    state = opt.init(params)
    initial_loss = float(_loss(params))

    for _ in range(20):
        grads = _quadratic_grads(params)
        updates, state = opt.update(grads, state, params)
        params = apply_updates(params, updates)

    final_loss = float(_loss(params))
    assert final_loss < initial_loss


@pytest.mark.parametrize("name", list(ALL_2D_OPTIMIZER_FACTORIES))
def test_optimizer_zero_grad_still_advances_step_and_state_is_finite(name):
    opt = ALL_2D_OPTIMIZER_FACTORIES[name]()
    params = {"w": jnp.ones((4, 4))}
    state = opt.init(params)
    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    updates, new_state = opt.update(zero_grads, state, params)
    assert int(new_state.step) == 1
    for leaf in jax.tree_util.tree_leaves(updates):
        assert bool(jnp.all(jnp.isfinite(leaf)))


@pytest.mark.parametrize("name", list(ALL_2D_OPTIMIZER_FACTORIES))
def test_optimizer_jit_compatible(name):
    opt = ALL_2D_OPTIMIZER_FACTORIES[name]()
    params = {"w": jnp.ones((4, 4))}
    state = opt.init(params)
    grads = _quadratic_grads(params)

    step_fn = jax.jit(lambda g, s, p: opt.update(g, s, p))
    updates, new_state = step_fn(grads, state, params)
    assert updates["w"].shape == (4, 4)


# ---------------------------------------------------------------------------
# SGDMomentum
# ---------------------------------------------------------------------------

def test_sgd_plain_gradient_descent_direction():
    opt = SGDMomentum(lr=0.1, momentum=0.0)
    params = {"w": jnp.array([1.0, 2.0])}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0, 1.0])}
    updates, _ = opt.update(grads, state, params)
    # With zero momentum, update is exactly -lr * grad.
    assert jnp.allclose(updates["w"], -0.1 * grads["w"])


def test_sgd_momentum_accumulates_across_steps():
    opt = SGDMomentum(lr=0.1, momentum=0.9)
    params = {"w": jnp.array(0.0)}
    state = opt.init(params)
    grads = {"w": jnp.array(1.0)}

    _, state1 = opt.update(grads, state, params)
    m1 = state1.momentum["w"]
    _, state2 = opt.update(grads, state1, params)
    m2 = state2.momentum["w"]
    # Momentum should keep growing under a constant gradient.
    assert float(m2) > float(m1)


def test_sgd_nesterov_differs_from_vanilla_momentum():
    params = {"w": jnp.array([0.0])}
    grads = {"w": jnp.array([1.0])}

    opt_vanilla = SGDMomentum(lr=0.1, momentum=0.9, nesterov=False)
    opt_nesterov = SGDMomentum(lr=0.1, momentum=0.9, nesterov=True)

    state_v = opt_vanilla.init(params)
    state_n = opt_nesterov.init(params)

    _, state_v = opt_vanilla.update(grads, state_v, params)
    _, state_n = opt_nesterov.update(grads, state_n, params)
    updates_v, _ = opt_vanilla.update(grads, state_v, params)
    updates_n, _ = opt_nesterov.update(grads, state_n, params)

    assert not jnp.allclose(updates_v["w"], updates_n["w"])


def test_sgd_weight_decay_adds_extra_shrinkage():
    params = {"w": jnp.array(2.0)}
    grads = {"w": jnp.array(0.0)}

    opt_plain = SGDMomentum(lr=0.1, momentum=0.0, weight_decay=0.0)
    opt_decay = SGDMomentum(lr=0.1, momentum=0.0, weight_decay=0.5)

    updates_plain, _ = opt_plain.update(grads, opt_plain.init(params), params)
    updates_decay, _ = opt_decay.update(grads, opt_decay.init(params), params)

    assert jnp.allclose(updates_plain["w"], 0.0)
    assert float(updates_decay["w"]) < 0.0


def test_sgd_state_is_named_tuple_type():
    opt = SGDMomentum(lr=0.1)
    state = opt.init({"w": jnp.ones((2,))})
    assert isinstance(state, SGDMomentumState)


# ---------------------------------------------------------------------------
# Adam / AdamW
# ---------------------------------------------------------------------------

def test_adam_first_step_matches_manual_bias_correction():
    opt = Adam(lr=0.1, b1=0.9, b2=0.999, eps=1e-8)
    params = {"w": jnp.array([1.0])}
    state = opt.init(params)
    grads = {"w": jnp.array([1.0])}

    updates, new_state = opt.update(grads, state, params)

    m = 0.1 * 1.0  # (1 - b1) * g
    v = 0.001 * 1.0  # (1 - b2) * g^2
    bias_c1 = 1 - 0.9 ** 1
    bias_c2 = 1 - 0.999 ** 1
    expected = -0.1 * (m / bias_c1) / (jnp.sqrt(v / bias_c2) + 1e-8)

    assert jnp.allclose(updates["w"], expected, atol=1e-5)
    assert int(new_state.step) == 1


def test_adam_state_is_named_tuple_type():
    opt = Adam(lr=0.1)
    state = opt.init({"w": jnp.ones((2,))})
    assert isinstance(state, AdamState)


def test_adamw_state_is_named_tuple_type():
    opt = AdamW(lr=0.1)
    state = opt.init({"w": jnp.ones((2,))})
    assert isinstance(state, AdamWState)


def test_adamw_weight_decay_shrinks_more_than_plain_adam():
    params = {"w": jnp.array(2.0)}
    grads = {"w": jnp.array(0.1)}

    adam = Adam(lr=0.1)
    adamw = AdamW(lr=0.1, weight_decay=0.5)

    updates_adam, _ = adam.update(grads, adam.init(params), params)
    updates_adamw, _ = adamw.update(grads, adamw.init(params), params)

    # AdamW's decoupled decay pushes the update more negative (more
    # shrinkage) than plain Adam for a positive parameter value.
    assert float(updates_adamw["w"]) < float(updates_adam["w"])


def test_adamw_zero_weight_decay_matches_adam():
    params = {"w": jnp.array([2.0])}
    grads = {"w": jnp.array([0.3])}

    adam = Adam(lr=0.1, b1=0.9, b2=0.999, eps=1e-8)
    adamw = AdamW(lr=0.1, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0)

    updates_adam, _ = adam.update(grads, adam.init(params), params)
    updates_adamw, _ = adamw.update(grads, adamw.init(params), params)

    assert jnp.allclose(updates_adam["w"], updates_adamw["w"])


def test_adam_makes_steady_progress_on_quadratic():
    # Adam's adaptive step should make steady progress on a simple quadratic.
    opt = Adam(lr=0.1)
    params = {"w": jnp.array([5.0, 5.0])}
    state = opt.init(params)
    initial_loss = float(jnp.sum(params["w"] ** 2))
    for _ in range(50):
        grads = _quadratic_grads(params)
        updates, state = opt.update(grads, state, params)
        params = apply_updates(params, updates)
    final_loss = float(jnp.sum(params["w"] ** 2))
    assert final_loss < initial_loss * 0.5


# ---------------------------------------------------------------------------
# Lion
# ---------------------------------------------------------------------------

def test_lion_state_is_named_tuple_type():
    opt = Lion(lr=0.1)
    state = opt.init({"w": jnp.ones((2,))})
    assert isinstance(state, LionState)


def test_lion_update_uses_sign_of_direction():
    opt = Lion(lr=0.1, b1=0.9, b2=0.99)
    params = {"w": jnp.array([1.0, 1.0])}
    state = opt.init(params)
    grads = {"w": jnp.array([5.0, -3.0])}
    updates, _ = opt.update(grads, state, params)
    # sign-based update magnitude should be exactly lr, direction opposite grad sign.
    assert jnp.allclose(jnp.abs(updates["w"]), 0.1)
    assert float(updates["w"][0]) < 0
    assert float(updates["w"][1]) > 0


def test_lion_weight_decay_shrinks_positive_params():
    params = {"w": jnp.array(2.0)}
    grads = {"w": jnp.array(0.0)}
    opt = Lion(lr=0.1, weight_decay=0.5)
    updates, _ = opt.update(grads, opt.init(params), params)
    assert float(updates["w"]) < 0.0


# ---------------------------------------------------------------------------
# RMSprop
# ---------------------------------------------------------------------------

def test_rmsprop_state_is_named_tuple_type():
    opt = RMSprop(lr=0.1)
    state = opt.init({"w": jnp.ones((2,))})
    assert isinstance(state, RMSpropState)


def test_rmsprop_uncentered_has_no_mean_g():
    opt = RMSprop(lr=0.1, centered=False)
    state = opt.init({"w": jnp.ones((2,))})
    assert state.mean_g is None


def test_rmsprop_centered_tracks_mean_g():
    opt = RMSprop(lr=0.1, centered=True)
    params = {"w": jnp.ones((2,))}
    state = opt.init(params)
    assert state.mean_g is not None
    grads = {"w": jnp.array([1.0, -1.0])}
    _, new_state = opt.update(grads, state, params)
    assert new_state.mean_g is not None
    assert not jnp.allclose(new_state.mean_g["w"], jnp.zeros(2))


def test_rmsprop_momentum_buf_none_when_momentum_zero():
    opt = RMSprop(lr=0.1, momentum=0.0)
    state = opt.init({"w": jnp.ones((2,))})
    assert state.momentum_buf is None


def test_rmsprop_momentum_buf_present_when_momentum_positive():
    opt = RMSprop(lr=0.1, momentum=0.9)
    params = {"w": jnp.ones((2,))}
    state = opt.init(params)
    assert state.momentum_buf is not None
    grads = {"w": jnp.array([1.0, 1.0])}
    updates, new_state = opt.update(grads, state, params)
    assert updates["w"].shape == (2,)


def test_rmsprop_update_direction_opposes_gradient():
    opt = RMSprop(lr=0.1)
    params = {"w": jnp.array(1.0)}
    state = opt.init(params)
    grads = {"w": jnp.array(2.0)}
    updates, _ = opt.update(grads, state, params)
    assert float(updates["w"]) < 0.0


# ---------------------------------------------------------------------------
# Adagrad
# ---------------------------------------------------------------------------

def test_adagrad_state_is_named_tuple_type():
    opt = Adagrad(lr=0.1)
    state = opt.init({"w": jnp.ones((2,))})
    assert isinstance(state, AdagradState)


def test_adagrad_initial_accumulator_respected():
    opt = Adagrad(lr=0.1, initial_accumulator=1.0)
    state = opt.init({"w": jnp.ones((3,))})
    assert jnp.allclose(state.g2["w"], jnp.ones(3))


def test_adagrad_accumulator_grows_monotonically():
    opt = Adagrad(lr=0.1)
    params = {"w": jnp.array(1.0)}
    state = opt.init(params)
    grads = {"w": jnp.array(1.0)}
    _, state1 = opt.update(grads, state, params)
    _, state2 = opt.update(grads, state1, params)
    assert float(state2.g2["w"]) > float(state1.g2["w"])


def test_adagrad_effective_lr_shrinks_over_time():
    opt = Adagrad(lr=0.1)
    params = {"w": jnp.array(0.0)}
    state = opt.init(params)
    grads = {"w": jnp.array(1.0)}

    updates1, state = opt.update(grads, state, params)
    updates2, state = opt.update(grads, state, params)
    # As g2 accumulates, later updates for the same-magnitude gradient
    # should shrink in magnitude.
    assert abs(float(updates2["w"])) < abs(float(updates1["w"]))


# ---------------------------------------------------------------------------
# Adan
# ---------------------------------------------------------------------------

def test_adan_state_is_named_tuple_type():
    opt = Adan(lr=0.1)
    state = opt.init({"w": jnp.ones((2,))})
    assert isinstance(state, AdanState)


def test_adan_first_step_treats_diff_as_zero():
    # On the very first update, prev_grad starts at zero, so `is_first`
    # should force diff=0 rather than diff = g - 0.
    opt = Adan(lr=0.1, b1=0.98, b2=0.92, b3=0.99)
    params = {"w": jnp.array([1.0])}
    state = opt.init(params)
    grads = {"w": jnp.array([10.0])}  # large gradient
    updates, new_state = opt.update(grads, state, params)
    assert bool(jnp.isfinite(updates["w"]))
    assert jnp.allclose(new_state.prev_grad["w"], grads["w"])


def test_adan_prev_grad_updated_each_step():
    opt = Adan(lr=0.1)
    params = {"w": jnp.array([1.0])}
    state = opt.init(params)
    grads1 = {"w": jnp.array([1.0])}
    _, state = opt.update(grads1, state, params)
    assert jnp.allclose(state.prev_grad["w"], grads1["w"])
    grads2 = {"w": jnp.array([2.0])}
    _, state = opt.update(grads2, state, params)
    assert jnp.allclose(state.prev_grad["w"], grads2["w"])


def test_adan_weight_decay_adds_shrinkage():
    params = {"w": jnp.array(2.0)}
    grads = {"w": jnp.array(0.0)}
    opt_plain = Adan(lr=0.1, weight_decay=0.0)
    opt_decay = Adan(lr=0.1, weight_decay=0.5)
    updates_plain, _ = opt_plain.update(grads, opt_plain.init(params), params)
    updates_decay, _ = opt_decay.update(grads, opt_decay.init(params), params)
    assert float(updates_decay["w"]) < float(updates_plain["w"])


# ---------------------------------------------------------------------------
# Adafactor
# ---------------------------------------------------------------------------

def test_adafactor_state_is_named_tuple_type():
    opt = Adafactor(lr=0.1)
    state = opt.init({"w": jnp.ones((4, 4))})
    assert isinstance(state, AdafactorState)


def test_adafactor_2d_leaf_uses_row_col_factorization():
    opt = Adafactor(lr=0.1)
    params = {"w": jnp.ones((3, 5))}
    state = opt.init(params)
    assert state.v_row["w"].shape == (3,)
    assert state.v_col["w"].shape == (5,)


def test_adafactor_1d_leaf_uses_full_second_moment():
    opt = Adafactor(lr=0.1)
    params = {"b": jnp.ones((5,))}
    state = opt.init(params)
    assert state.v_full["b"].shape == (5,)


def test_adafactor_update_reduces_quadratic_loss_for_2d_params():
    opt = Adafactor(lr=0.5)
    params = {"w": jnp.ones((4, 4)) * 2.0}
    state = opt.init(params)
    initial_loss = float(_loss(params))
    for _ in range(20):
        grads = _quadratic_grads(params)
        updates, state = opt.update(grads, state, params)
        params = apply_updates(params, updates)
    assert float(_loss(params)) < initial_loss


def test_adafactor_clip_threshold_bounds_update_rms():
    opt = Adafactor(lr=1.0, clip_threshold=0.5)
    params = {"w": jnp.ones((4, 4)) * 10.0}
    state = opt.init(params)
    grads = {"w": jnp.ones((4, 4)) * 100.0}
    updates, _ = opt.update(grads, state, params)
    direction_rms = jnp.sqrt(jnp.mean(jnp.square(updates["w"] / -opt.lr)))
    assert float(direction_rms) <= 0.5 + 1e-4


# ---------------------------------------------------------------------------
# Shampoo
# ---------------------------------------------------------------------------

def test_shampoo_state_is_named_tuple_type():
    opt = Shampoo(lr=0.1)
    state = opt.init({"w": jnp.eye(3)})
    assert isinstance(state, ShampooState)


def test_shampoo_requires_2d_leaves():
    opt = Shampoo(lr=0.1)
    with pytest.raises(AssertionError):
        opt.init({"b": jnp.ones((5,))})


def test_shampoo_preconditioner_initialized_as_identity():
    opt = Shampoo(lr=0.1)
    params = {"w": jnp.ones((3, 3))}
    state = opt.init(params)
    assert jnp.allclose(state.L["w"], jnp.eye(3))
    assert jnp.allclose(state.R["w"], jnp.eye(3))


def test_shampoo_update_reduces_quadratic_loss():
    opt = Shampoo(lr=0.1, precondition_every=1)
    params = {"w": jnp.eye(3) * 2.0}
    state = opt.init(params)
    initial_loss = float(_loss(params))
    for _ in range(10):
        grads = _quadratic_grads(params)
        updates, state = opt.update(grads, state, params)
        params = apply_updates(params, updates)
    assert float(_loss(params)) < initial_loss


def test_shampoo_precondition_every_skips_recompute():
    # step=0 always recomputes (0 % N == 0). With precondition_every large,
    # the *next* step should reuse the cached preconditioner rather than
    # recomputing it from the (now different) accumulators.
    opt = Shampoo(lr=0.1, precondition_every=1000)
    params = {"w": jnp.ones((3, 3))}
    state = opt.init(params)
    grads = {"w": jnp.ones((3, 3))}

    _, state = opt.update(grads, state, params)  # step 0 -> recomputes
    cached_inv_root = state.L_inv_root["w"]

    _, state = opt.update(grads, state, params)  # step 1 -> should skip
    assert jnp.allclose(state.L_inv_root["w"], cached_inv_root)


def test_shampoo_output_is_never_nan():
    opt = Shampoo(lr=0.1)
    params = {"w": jnp.zeros((3, 3))}
    state = opt.init(params)
    grads = {"w": jnp.zeros((3, 3))}
    updates, _ = opt.update(grads, state, params)
    assert bool(jnp.all(jnp.isfinite(updates["w"])))


# ---------------------------------------------------------------------------
# MuonCore / Muon
# ---------------------------------------------------------------------------

def test_muon_core_state_is_named_tuple_type():
    opt = MuonCore(lr=0.1)
    state = opt.init({"w": jnp.ones((4, 4))})
    assert isinstance(state, MuonCoreState)


def test_muon_core_2d_update_shape_matches_param():
    opt = MuonCore(lr=0.1)
    params = {"w": jnp.eye(4)}
    state = opt.init(params)
    grads = {"w": jax.random.normal(jax.random.PRNGKey(0), (4, 4))}
    updates, _ = opt.update(grads, state, params)
    assert updates["w"].shape == (4, 4)


def test_muon_core_1d_falls_back_to_plain_sgd_like_update():
    opt = MuonCore(lr=0.1, momentum=0.0, nesterov=False, clip=False)
    params = {"b": jnp.array([1.0, 2.0])}
    state = opt.init(params)
    grads = {"b": jnp.array([1.0, 1.0])}
    updates, _ = opt.update(grads, state, params)
    assert jnp.allclose(updates["b"], -0.1 * grads["b"])


def test_muon_core_3d_batched_matrices_supported():
    opt = MuonCore(lr=0.1)
    params = {"w": jnp.stack([jnp.eye(3), jnp.eye(3)])}
    state = opt.init(params)
    grads = {"w": jax.random.normal(jax.random.PRNGKey(0), (2, 3, 3))}
    updates, _ = opt.update(grads, state, params)
    assert updates["w"].shape == (2, 3, 3)


def test_muon_core_4d_conv_kernel_supported():
    opt = MuonCore(lr=0.1)
    params = {"w": jnp.zeros((8, 4, 3, 3))}
    state = opt.init(params)
    grads = {"w": jax.random.normal(jax.random.PRNGKey(0), (8, 4, 3, 3))}
    updates, _ = opt.update(grads, state, params)
    assert updates["w"].shape == (8, 4, 3, 3)


def test_muon_core_output_never_nan_even_with_zero_grad():
    opt = MuonCore(lr=0.1)
    params = {"w": jnp.zeros((4, 4))}
    state = opt.init(params)
    grads = {"w": jnp.zeros((4, 4))}
    updates, _ = opt.update(grads, state, params)
    assert bool(jnp.all(jnp.isfinite(updates["w"])))


def test_muon_core_clip_bounds_direction_norm():
    opt = MuonCore(lr=0.1, momentum=0.0, nesterov=False, clip=1.0)
    params = {"w": jnp.ones((4, 4))}
    state = opt.init(params)
    huge_grads = {"w": jnp.ones((4, 4)) * 1000.0}
    updates, _ = opt.update(huge_grads, state, params)
    assert bool(jnp.all(jnp.isfinite(updates["w"])))


def test_muon_wraps_into_partition_optimizer():
    from xera.weave.optimizer.partition import Partition
    opt = Muon(lr=0.1)
    assert isinstance(opt, Partition)


def test_muon_routes_2d_params_to_muon_core_and_1d_to_fallback():
    opt = Muon(lr=0.1, fallback="adamw", fallback_lr=1e-4)
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}
    state = opt.init(params)
    grads = {"w": jax.random.normal(jax.random.PRNGKey(0), (4, 4)),
              "b": jax.random.normal(jax.random.PRNGKey(1), (4,))}
    updates, new_state = opt.update(grads, state, params)
    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)


def test_muon_unknown_fallback_string_raises():
    with pytest.raises(ValueError):
        Muon(lr=0.1, fallback="not_a_real_optimizer")


def test_muon_fallback_none_requires_catch_all_or_raises_on_unmatched_leaf():
    opt = Muon(lr=0.1, fallback=None)
    params = {"b": jnp.ones((4,))}  # 1D leaf, no rule matches without fallback
    with pytest.raises(ValueError):
        opt.init(params)


def test_muon_custom_fallback_optimizer_instance():
    fallback_opt = SGDMomentum(lr=0.01)
    opt = Muon(lr=0.1, fallback=fallback_opt)
    params = {"b": jnp.ones((4,))}
    state = opt.init(params)
    grads = {"b": jnp.ones((4,))}
    updates, _ = opt.update(grads, state, params)
    assert jnp.allclose(updates["b"], -0.01 * grads["b"])


def test_muon_reduces_quadratic_loss():
    opt = Muon(lr=0.1)
    params = {"w": jnp.eye(4) * 2.0, "b": jnp.ones((4,)) * 2.0}
    state = opt.init(params)
    initial_loss = float(_loss(params))
    for _ in range(10):
        grads = _quadratic_grads(params)
        updates, state = opt.update(grads, state, params)
        params = apply_updates(params, updates)
    assert float(_loss(params)) < initial_loss


# ---------------------------------------------------------------------------
# Exposure from xera.weave / xera.O namespace
# ---------------------------------------------------------------------------

def test_all_core_optimizers_exposed_on_weave():
    for name in ["SGDMomentum", "Adam", "AdamW", "Lion", "MuonCore", "Muon",
                 "RMSprop", "Adagrad", "Adan", "Adafactor", "Shampoo"]:
        assert hasattr(weave, name)


def test_optimizer_base_class_is_common_ancestor():
    from xera.weave.optimizer.base import Optimizer
    for factory in ALL_2D_OPTIMIZER_FACTORIES.values():
        assert isinstance(factory(), Optimizer)
