"""
Loop: a plain function for running JAX-based training/inference iterations.

`loop` is intentionally not a class -- it takes a `body_fn` and returns
`(final_carry, outputs)`, nothing more. It composes with everything else
in `weave` (a `Trainer(Struct).run()` calls it; `Callback.print`/`Callback.io`
are called from inside the `body_fn` you pass it; a stop-condition like
`Callback.early_stopping`/`Callback.nan` is passed as `stop=`) without any
of them needing to know about each other.

## The `stop=` branch

When `stop` is given, `loop` always builds two branches under the hood:
your `body_fn` (the real step), and an internal, generic dummy branch
that passes the carry through unchanged and emits zeroed/masked output
of the same shape. Every step, `stop(carry, x)` is called and OR'd into
a `stopped` flag carried alongside your own carry; `jax.lax.cond` picks
`body_fn` or the dummy branch based on that flag. Once `stopped` is
`True` it can never go back to `False` -- there's no code path back to
it -- so once a stop condition fires, every remaining step (up to
`steps`) takes the cheap dummy path. The scan/loop itself is still run
for the full `steps` (jit needs a static trip count), only what happens
per remaining step becomes cheap.

`stop` can be anything callable as `stop(carry, x) -> bool`; `loop`
doesn't care what it is (`Callback.early_stopping(...)`,
`Callback.nan(...)`, or a plain function all work the same way).
"""

from __future__ import annotations
import jax
import jax.numpy as jnp


def loop(body_fn, init_carry, xs=None, type="scan", steps=1000, stop=None):
    """
    Run an iterative computation with `jax.lax.scan` or `jax.lax.fori_loop`.

    Args:
        body_fn: A function `(carry, x) -> (new_carry, output)`, called
            once per step (subject to `stop`, see above).
        init_carry: The initial carry state passed to the first iteration.
        xs: Optional sequence of per-step inputs. If `None`, defaults to
            `jnp.arange(steps)`.
        type: `"scan"` (default, via `jax.lax.scan`) or `"fori_loop"`
            (via `jax.lax.fori_loop`, with outputs collected manually).
        steps: Number of iterations. Static -- fixed regardless of `stop`.
        stop: Optional `stop(carry, x) -> bool`. When given, `loop` wraps
            `body_fn` so that once `stop` first returns `True`, every
            following step takes a cheap, generic dummy branch instead
            of calling `body_fn` (see module docstring). When `None`
            (default), `body_fn` runs every step, exactly as before.

    Returns:
        A tuple `(final_carry, outputs)`.

    Example:
        >>> def step(carry, x):
        ...     return carry + x, carry * x
        >>> final_carry, outputs = loop(step, init_carry=0, steps=5,
        ...                              xs=jnp.array([1, 2, 3, 4, 5]))

        >>> # With a stop condition:
        >>> final, outputs = loop(train_step, init_carry=state, steps=1000,
        ...                        stop=Callback.early_stopping(patience=10))
    """
    assert type in ("fori_loop", "scan"), f"unknown loop type: {type}"

    if xs is None:
        xs = jnp.arange(steps)

    run_body_fn, run_init_carry, unwrap = _prepare(body_fn, init_carry, xs, stop)

    if type == "fori_loop":
        final_carry, outputs = _run_fori(run_body_fn, run_init_carry, xs, steps)
    else:
        final_carry, outputs = jax.lax.scan(run_body_fn, run_init_carry, xs, length=steps)

    return unwrap(final_carry), outputs


def _prepare(body_fn, init_carry, xs, stop):
    """
    Build the actual per-step function `loop` runs, plus a matching
    initial carry and an `unwrap` to strip any bookkeeping back off the
    final carry before returning it to the caller.

    When `stop is None` this is the identity: `body_fn`/`init_carry`
    pass straight through, nothing is added.

    When `stop` is given, the carry becomes `(stopped, user_carry)`, and
    the per-step function:
      1. runs `stop(user_carry, x)` and ORs it into `stopped`
      2. picks `body_fn` or a generic dummy via `jax.lax.cond` on the
         (already-updated) `stopped` flag
    The dummy branch is only built once real output shapes are known,
    via a shape-only trace of `body_fn` on `init_carry`/`x0` -- this
    costs no extra compute at run time, it only fixes the dummy's
    output pytree structure/dtype so `cond`'s two branches match.
    """
    if stop is None:
        return body_fn, init_carry, lambda c: c

    x0 = jax.tree_util.tree_map(lambda a: a[0], xs)
    _, sample_output = jax.eval_shape(body_fn, init_carry, x0)
    dummy_output = jax.tree_util.tree_map(
        lambda s: jnp.zeros(s.shape, dtype=s.dtype), sample_output
    )

    def dummy_fn(carry, x):
        return carry, dummy_output

    def wrapped(carry, x):
        stopped, user_carry = carry
        stopped = jnp.logical_or(stopped, jnp.asarray(stop(user_carry, x), dtype=bool))
        new_user_carry, output = jax.lax.cond(
            stopped, dummy_fn, body_fn, user_carry, x
        )
        return (stopped, new_user_carry), output

    wrapped_init_carry = (jnp.asarray(False), init_carry)
    return wrapped, wrapped_init_carry, lambda c: c[1]


def _run_fori(body_fn, init_carry, xs, steps):
    """`fori_loop` execution path, mirroring `jax.lax.scan`'s xs/output handling."""
    x0 = jax.tree_util.tree_map(lambda a: a[0], xs)
    _, sample_output = body_fn(init_carry, x0)
    sample_output = jax.tree_util.tree_map(jnp.asarray, sample_output)

    outputs = jax.tree_util.tree_map(
        lambda s: jnp.zeros((steps,) + s.shape, dtype=s.dtype), sample_output
    )

    def fori_body(i, carry):
        original_carry, outputs_array = carry
        x_i = jax.tree_util.tree_map(lambda a: a[i], xs)
        new_carry, output = body_fn(original_carry, x_i)
        new_outputs = jax.tree_util.tree_map(
            lambda arr, out: arr.at[i].set(out), outputs_array, output
        )
        return (new_carry, new_outputs)

    return jax.lax.fori_loop(0, steps, fori_body, (init_carry, outputs))


__all__ = ["loop"]
