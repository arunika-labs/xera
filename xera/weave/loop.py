

from __future__ import annotations
import jax
import jax.numpy as jnp
from .state import State


class Loop(State):
    """Thin wrapper over `lax.scan`/`lax.fori_loop`. Composes naturally
    for things like nested/grad-accumulation loops -- give the outer
    class its own `Loop` and the inner class its own `Loop`, rather than
    threading one `Loop` through both (no special "nested" mode needed
    here, unlike an earlier version of this file).

    `type="scan"` (default) collects a stacked per-step output `ys`
    alongside the final carry -- what `Train` relies on for its returned
    `losses`. `type="fori_loop"` returns only the final carry (no `ys`,
    `lax.fori_loop` doesn't collect per-step outputs at all) -- use it
    directly for loops that don't need a per-step history, not through
    `Train` (which always wants `ys`; see `Train`'s `loop_type` assert).
    """

    type: str = "scan"
    steps: int = 1

    def setup(self):
        assert self.type in ("fori_loop", "scan"), f"unknown loop type: {self.type}"

    def run(self, body_fn, init_carry, xs=None):
        if self.type == "fori_loop":
            # lax.fori_loop always calls body_fun(i, val) -- body_fn here
            # follows scan's (carry, x) convention instead, so adapt here
            # rather than asking every body_fn to support both orders.
            final_carry = jax.lax.fori_loop(
                0, self.steps, lambda i, carry: body_fn(carry, i), init_carry
            )
            return final_carry, None

        if xs is None:
            xs = jnp.arange(self.steps)
        final_carry, ys = jax.lax.scan(body_fn, init_carry, xs, length=self.steps)
        return final_carry, ys


__all__ = ["Loop"]
