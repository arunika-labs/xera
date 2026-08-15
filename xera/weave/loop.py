

from __future__ import annotations
import jax
import jax.numpy as jnp
from .state import State


class Loop(State):
    
    type: str = "scan"     
    steps: int = 1
    inner: "Loop" = None   

    def setup(self):
        assert self.type in ("fori_loop", "scan"), f"unknown loop type: {self.type}"

    def run(self, body_fn, init_carry, xs=None):
        
        if self.inner is not None:
            return self._run_nested(body_fn, init_carry, xs)

        if self.type == "fori_loop":
            def fori_body(i, carry):
                return body_fn(i, carry)
            return jax.lax.fori_loop(0, self.steps, fori_body, init_carry)

        
        if xs is None:
            
            xs = jnp.arange(self.steps)
        final_carry, ys = jax.lax.scan(body_fn, init_carry, xs, length=self.steps)
        return final_carry, ys

    def _run_nested(self, body_fn, init_carry, xs):
        
        if self.type == "fori_loop":
            def fori_body(i, carry):
                return body_fn(carry, i)
            return jax.lax.fori_loop(0, self.steps, fori_body, init_carry)

        if xs is None:
            xs = jnp.arange(self.steps)
        final_carry, ys = jax.lax.scan(body_fn, init_carry, xs, length=self.steps)
        return final_carry, ys


__all__ = ["Loop"]