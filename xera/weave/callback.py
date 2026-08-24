"""
Callback: small, composable pieces used from inside a training step.

`Callback` has no state and no class hierarchy to subclass -- it's a
namespace of two kinds of plain functions, split by what they're for:

- **Side-effects** (`Callback.print`, `Callback.io`): called from *inside*
  the `body_fn` you write, to do something outside the traced computation
  each step (print a metric, write a checkpoint). They don't affect
  `weave.loop`'s control flow at all.

- **Stop conditions** (`Callback.early_stopping`, `Callback.nan`): these
  are *factories* -- calling them returns a `stop_fn(carry, x) -> bool`
  meant to be passed as `weave.loop(..., stop=stop_fn)`. `loop` itself
  owns the two-branch (real step / cheap dummy) mechanism and the sticky
  latch (see `xera.weave.loop`); a stop condition only ever needs to
  answer "should we be stopped as of this step?".

Nothing here is a `Struct`/pytree -- these are just Python functions
composed together by whatever calls them, by design.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp


class Callback:
    """Namespace for step side-effects and loop stop-conditions."""

    @staticmethod
    def print(step, **values):
        """
        Print step metrics from inside a traced `body_fn`, via
        `jax.debug.print` (safe to call under `jit`/`scan`).

        Args:
            step: The current step (usually the scan/fori index).
            **values: Named values to print alongside the step, e.g.
                `loss=loss, lr=current_lr`.

        Example:
            >>> def body_fn(carry, i):
            ...     ...
            ...     Callback.print(i, loss=loss)
            ...     return new_carry, loss
        """
        parts = " ".join(f"{name}={{{name}}}" for name in values)
        jax.debug.print("step={step} " + parts, step=step, **values)

    @staticmethod
    def io(step, fn, *args, **kwargs):
        """
        Run an arbitrary Python side-effect (file I/O, checkpointing,
        external logging, ...) from inside a traced `body_fn`, via
        `jax.experimental.io_callback`.

        Args:
            step: The current step (passed through to `fn` as the first
                argument, so `fn` can name checkpoint files, etc.).
            fn: A plain Python function `fn(step, *args, **kwargs)`. Its
                return value is discarded (`result_shape_dtypes=None`);
                use `io` for effects, not for feeding values back into
                the traced computation.
            *args, **kwargs: Forwarded to `fn`.

        Example:
            >>> def _checkpoint(step, model, optimizer, meta):
            ...     model.save_struct(model, optimizer, meta, f"ckpt_{int(step)}.sxera")
            ...
            >>> def body_fn(carry, i):
            ...     model, opt_state = carry
            ...     ...
            ...     Callback.io(i, _checkpoint, model, opt_state, {"step": i})
            ...     return (new_model, new_opt_state), loss
        """
        jax.experimental.io_callback(fn, None, step, *args, **kwargs, ordered=True)

    @staticmethod
    def early_stopping(patience, extract):
        """
        Build a `stop_fn(carry, x) -> bool` for `weave.loop(..., stop=...)`
        that fires once a tracked value hasn't improved for `patience`
        consecutive steps.

        Args:
            patience: Number of consecutive non-improving steps to
                tolerate before signaling stop.
            extract: `extract(carry) -> since_improved`. Since `loop`'s
                stop condition is a pure function with no state of its
                own, "steps since improvement" needs to live somewhere
                -- typically as part of your own carry, incremented or
                reset to 0 each step in `body_fn` depending on whether
                that step improved (comparison direction, e.g. `min` vs
                `max`, belongs entirely to that update rule). `extract`
                tells this factory how to read that counter back out of
                whatever carry shape you're using.

        Returns:
            `stop_fn(carry, x) -> bool`.

        Example:
            >>> # carry = (model, opt_state, best_loss, since_improved)
            >>> stop = Callback.early_stopping(
            ...     patience=10, extract=lambda carry: carry[3],
            ... )
            >>> final, outputs = loop(train_step, init_carry=carry0,
            ...                        steps=1000, stop=stop)
        """
        def stop_fn(carry, x):
            return jnp.asarray(extract(carry)) >= patience

        return stop_fn

    @staticmethod
    def nan():
        """
        Build a `stop_fn(carry, x) -> bool` for `weave.loop(..., stop=...)`
        that fires once any array leaf in `carry` is `NaN` or `Inf`.

        Like any `stop=` condition, this is checked against the
        *pre-step* carry, before `body_fn` runs -- so a NaN first
        produced at step N is detected at step N+1 and latches from
        there (the same one-step-delay behavior as periodic checks like
        `Callback.early_stopping`). The single step that produced the
        NaN still reaches `outputs`; every step after that takes the
        cheap dummy branch.

        Returns:
            `stop_fn(carry, x) -> bool`.

        Example:
            >>> final, outputs = loop(train_step, init_carry=state,
            ...                        steps=1000, stop=Callback.nan())
        """
        def stop_fn(carry, x):
            leaves = jax.tree_util.tree_leaves(carry)
            bad = jnp.asarray(False)
            for leaf in leaves:
                if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
                    bad = jnp.logical_or(bad, jnp.logical_not(jnp.all(jnp.isfinite(leaf))))
            return bad

        return stop_fn


__all__ = ["Callback"]
