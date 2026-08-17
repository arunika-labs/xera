

from __future__ import annotations
from typing import Any, Callable, Sequence, Tuple
import jax
from .base import Optimizer

Predicate = Callable[[Any, Any], bool]


class PartitionState:
    __slots__ = ("inner_states", "assignment")

    def __init__(self, inner_states, assignment):
        self.inner_states = inner_states
        self.assignment = assignment

    def __repr__(self):
        return f"PartitionState(assignment={self.assignment!r})"


jax.tree_util.register_pytree_node(
    PartitionState,
    lambda s: (s.inner_states, s.assignment),
    lambda assignment, children: PartitionState(children, assignment),
)


class Partition(Optimizer):


    def __init__(self, rules: Sequence[Tuple[Predicate, Optimizer]]):
        assert len(rules) > 0, "Partition needs at least one (predicate, optimizer) rule"
        self.rules = list(rules)

    def _assign(self, params):
        path_leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
        assignment = []
        for path, leaf in path_leaves:
            idx = None
            for i, (pred, _opt) in enumerate(self.rules):
                if pred(path, leaf):
                    idx = i
                    break
            if idx is None:
                raise ValueError(
                    f"Partition: no rule matched leaf at path {path!r} "
                    f"(shape={getattr(leaf, 'shape', None)}). Add a "
                    "catch-all rule (e.g. `lambda path, leaf: True`) as "
                    "the last entry, or narrow your other predicates."
                )
            assignment.append(idx)
        return treedef, tuple(assignment)

    def init(self, params):
        _treedef, assignment = self._assign(params)
        path_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
        leaves = [l for _, l in path_leaves]

        inner_states = []
        for i, (_pred, opt) in enumerate(self.rules):
            group = tuple(l for l, a in zip(leaves, assignment) if a == i)
            inner_states.append(opt.init(group))
        return PartitionState(inner_states=tuple(inner_states), assignment=assignment)

    def update(self, grads, state, params=None, step=None):
        path_leaves, treedef = jax.tree_util.tree_flatten_with_path(grads)
        grad_leaves = [l for _, l in path_leaves]
        param_leaves = (
            treedef.flatten_up_to(params) if params is not None
            else [None] * len(grad_leaves)
        )

        assignment = state.assignment
        out = [None] * len(grad_leaves)
        new_inner_states = list(state.inner_states)

        for i, (_pred, opt) in enumerate(self.rules):
            idx = [j for j, a in enumerate(assignment) if a == i]
            group_grads = tuple(grad_leaves[j] for j in idx)
            group_params = (
                tuple(param_leaves[j] for j in idx) if params is not None else None
            )
            group_updates, new_inner_states[i] = opt.update(
                group_grads, state.inner_states[i], group_params, step
            )
            for j, u in zip(idx, group_updates):
                out[j] = u

        updates = jax.tree_util.tree_unflatten(treedef, out)
        return updates, PartitionState(
            inner_states=tuple(new_inner_states), assignment=assignment
        )


__all__ = ["Partition", "PartitionState", "Predicate"]
