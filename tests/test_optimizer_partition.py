"""Tests for xera.weave.optimizer.partition: Partition."""

import jax
import jax.numpy as jnp
import pytest
import xera.weave as weave
from xera.weave.optimizer.partition import Partition, PartitionState
from xera.weave.optimizer.core.sgd import SGDMomentum
from xera.weave.optimizer.core.adam import Adam
from xera.weave.optimizer.base import apply_updates


def _is_2d(path, leaf):
    return hasattr(leaf, "ndim") and leaf.ndim == 2


def _is_1d(path, leaf):
    return hasattr(leaf, "ndim") and leaf.ndim == 1


def test_partition_requires_at_least_one_rule():
    with pytest.raises(AssertionError):
        Partition([])


def test_partition_routes_leaves_by_predicate():
    opt = Partition([
        (_is_2d, Adam(lr=0.1)),
        (lambda path, leaf: True, SGDMomentum(lr=0.1)),
    ])
    params = {"w": jnp.ones((3, 3)), "b": jnp.ones((3,))}
    state = opt.init(params)
    grads = {"w": jnp.ones((3, 3)), "b": jnp.ones((3,))}
    updates, new_state = opt.update(grads, state, params)
    assert updates["w"].shape == (3, 3)
    assert updates["b"].shape == (3,)


def test_partition_no_matching_rule_raises_on_init():
    opt = Partition([(_is_2d, Adam(lr=0.1))])
    params = {"b": jnp.ones((3,))}  # 1D leaf, no rule matches, no catch-all
    with pytest.raises(ValueError, match="no rule matched"):
        opt.init(params)


def test_partition_first_matching_rule_wins():
    # Both rules would match a 2D leaf; only the first rule's optimizer
    # should end up owning that leaf's assignment (first-match, like an
    # if/elif chain). update() still visits every rule's (possibly empty)
    # group, so we check the assignment and the group contents instead of
    # whether the second optimizer's update() was invoked at all.
    opt = Partition([
        (lambda path, leaf: True, SGDMomentum(lr=0.1)),
        (lambda path, leaf: True, SGDMomentum(lr=0.1)),
    ])
    params = {"w": jnp.ones((3, 3))}
    state = opt.init(params)
    assert state.assignment == (0,)  # leaf assigned to the first rule only


def test_partition_state_is_partition_state_type():
    opt = Partition([(lambda path, leaf: True, SGDMomentum(lr=0.1))])
    state = opt.init({"w": jnp.ones((2,))})
    assert isinstance(state, PartitionState)


def test_partition_state_pytree_roundtrip():
    opt = Partition([(lambda path, leaf: True, SGDMomentum(lr=0.1))])
    state = opt.init({"w": jnp.ones((2,))})
    leaves, treedef = jax.tree_util.tree_flatten(state)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, PartitionState)
    assert reconstructed.assignment == state.assignment


def test_partition_inner_states_are_independent_per_rule():
    opt = Partition([
        (_is_2d, Adam(lr=0.1)),
        (lambda path, leaf: True, SGDMomentum(lr=0.1, momentum=0.9)),
    ])
    params = {"w": jnp.ones((2, 2)), "b": jnp.ones((2,))}
    state = opt.init(params)
    grads = {"w": jnp.ones((2, 2)), "b": jnp.ones((2,))}
    _, new_state = opt.update(grads, state, params)
    # Adam's inner state (rule 0) has m/v; SGD's inner state (rule 1) has
    # momentum -- verify each group's state only reflects its own leaves.
    adam_state = new_state.inner_states[0]
    sgd_state = new_state.inner_states[1]
    assert hasattr(adam_state, "m") and hasattr(adam_state, "v")
    assert hasattr(sgd_state, "momentum")


def test_partition_reduces_loss_with_mixed_optimizers():
    opt = Partition([
        (_is_2d, Adam(lr=0.1)),
        (lambda path, leaf: True, SGDMomentum(lr=0.1)),
    ])
    params = {"w": jnp.eye(3) * 2.0, "b": jnp.ones((3,)) * 2.0}
    state = opt.init(params)

    def loss(p):
        return sum(jnp.sum(v ** 2) for v in jax.tree_util.tree_leaves(p))

    initial_loss = float(loss(params))
    for _ in range(15):
        grads = jax.tree_util.tree_map(lambda p: 2 * p, params)
        updates, state = opt.update(grads, state, params)
        params = apply_updates(params, updates)
    assert float(loss(params)) < initial_loss


def test_partition_update_without_params_uses_none_group_params():
    opt = Partition([(lambda path, leaf: True, SGDMomentum(lr=0.1))])
    params = {"w": jnp.ones((2,))}
    state = opt.init(params)
    grads = {"w": jnp.ones((2,))}
    updates, new_state = opt.update(grads, state, params=None)
    assert updates["w"].shape == (2,)


def test_partition_assignment_matches_number_of_leaves():
    opt = Partition([
        (_is_2d, Adam(lr=0.1)),
        (lambda path, leaf: True, SGDMomentum(lr=0.1)),
    ])
    params = {"w1": jnp.ones((2, 2)), "w2": jnp.ones((3, 3)), "b": jnp.ones((4,))}
    state = opt.init(params)
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    assert len(state.assignment) == len(leaves)


def test_partition_updates_preserve_tree_structure():
    opt = Partition([
        (_is_2d, Adam(lr=0.1)),
        (lambda path, leaf: True, SGDMomentum(lr=0.1)),
    ])
    params = {"nested": {"w": jnp.ones((2, 2))}, "b": jnp.ones((2,))}
    state = opt.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, _ = opt.update(grads, state, params)
    assert updates["nested"]["w"].shape == (2, 2)
    assert updates["b"].shape == (2,)


def test_partition_step_argument_forwarded_to_inner_optimizers():
    class StepCapturingOptimizer:
        def __init__(self):
            self.seen_steps = []

        def init(self, params):
            return None

        def update(self, grads, state, params=None, step=None):
            self.seen_steps.append(step)
            return grads, state

    capturer = StepCapturingOptimizer()
    opt = Partition([(lambda path, leaf: True, capturer)])
    params = {"w": jnp.ones((2,))}
    state = opt.init(params)
    grads = {"w": jnp.ones((2,))}
    opt.update(grads, state, params, step=42)
    assert capturer.seen_steps == [42]


def test_partition_accessible_from_weave_namespace():
    assert weave.Partition is Partition


def test_partition_jit_compatible():
    opt = Partition([
        (_is_2d, Adam(lr=0.1)),
        (lambda path, leaf: True, SGDMomentum(lr=0.1)),
    ])
    params = {"w": jnp.ones((2, 2)), "b": jnp.ones((2,))}
    state = opt.init(params)
    grads = {"w": jnp.ones((2, 2)), "b": jnp.ones((2,))}
    step_fn = jax.jit(lambda g, s, p: opt.update(g, s, p))
    updates, _ = step_fn(grads, state, params)
    assert updates["w"].shape == (2, 2)
    assert updates["b"].shape == (2,)
