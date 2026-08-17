

"""
Loop module for JAX-based training loops.

This module provides a flexible loop abstraction that supports both
`jax.lax.scan` and `jax.lax.fori_loop` implementations for training
and inference iterations.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from .state import State


class Loop(State):
    """
    A flexible loop abstraction for JAX-based training and inference.
    
    This class provides a unified interface for running iterative computations
    using either `jax.lax.scan` (default) or `jax.lax.fori_loop`. The loop
    maintains a carry state that can be updated across iterations and can
    optionally collect outputs from each iteration.
    
    Attributes:
        type: The type of loop to use. Either "scan" (default) or "fori_loop".
            - "scan": Uses jax.lax.scan for efficient compiled loops
            - "fori_loop": Uses jax.lax.fori_loop with explicit output collection
        steps: The number of iterations to run the loop.
    
    Example:
        >>> loop = Loop(type="scan", steps=100)
        >>> def body_fn(carry, i):
        ...     new_carry = carry + 1
        ...     return new_carry, carry  # return (new_carry, output)
        >>> final_carry, outputs = loop.run(body_fn, init_carry=0)
    """

    type: str = "scan"
    steps: int = 1

    def setup(self):
        """
        Validate the loop type configuration.
        
        Raises:
            AssertionError: If the loop type is not "fori_loop" or "scan".
        """
        assert self.type in ("fori_loop", "scan"), f"unknown loop type: {self.type}"

    def run(self, body_fn, init_carry, xs=None):
        """
        Execute the loop with the given body function.
        
        Args:
            body_fn: A function that takes (carry, i) or (carry, x) and returns
                (new_carry, output). For "scan" type with xs provided, it takes
                (carry, x) where x is from the xs sequence. Otherwise it takes
                (carry, i) where i is the iteration index.
            init_carry: The initial carry state passed to the first iteration.
            xs: Optional sequence of inputs for each iteration. If None and
                type is "scan", uses jnp.arange(steps) as the sequence.
        
        Returns:
            A tuple (final_carry, outputs) where:
                - final_carry: The carry state after the final iteration
                - outputs: Collected outputs from each iteration
        
        Example:
            >>> loop = Loop(type="scan", steps=5)
            >>> def step(carry, x):
            ...     return carry + x, carry * x
            >>> final_carry, outputs = loop.run(step, init_carry=0, xs=jnp.array([1,2,3,4,5]))
        """
        if self.type == "fori_loop":
            # fori_loop can collect outputs by pre-allocating an array
            # and filling it manually in the loop. The carry includes both
            # the original carry state and the output array.
            
            # Run the body function once to determine output shape/dtype
            _, sample_output = body_fn(init_carry, 0)
            
            # Pre-allocate output array
            outputs = jnp.zeros((self.steps,) + sample_output.shape, dtype=sample_output.dtype)
            
            def fori_body(i, carry):
                """
                Body function for fori_loop that maintains both carry and outputs.
                
                Args:
                    i: Current iteration index
                    carry: Tuple of (original_carry, outputs_array)
                
                Returns:
                    Updated tuple of (new_carry, updated_outputs_array)
                """
                original_carry, outputs_array = carry
                new_carry, output = body_fn(original_carry, i)
                # Update the output array at position i
                new_outputs = outputs_array.at[i].set(output)
                return (new_carry, new_outputs)
            
            final_carry, final_outputs = jax.lax.fori_loop(
                0, self.steps, fori_body, (init_carry, outputs)
            )
            return final_carry, final_outputs

        # Use jax.lax.scan for efficient compiled looping
        if xs is None:
            xs = jnp.arange(self.steps)
        final_carry, ys = jax.lax.scan(body_fn, init_carry, xs, length=self.steps)
        return final_carry, ys


__all__ = ["Loop"]
