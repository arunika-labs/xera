"""
Callback: small, composable pieces used from inside a training step.

`Callback` has no state and no class hierarchy to subclass -- it's a
namespace of a few kinds of plain functions, split by what they're for:

- **Side-effects** (`Callback.print`, `Callback.io`): called from *inside*
  the `body_fn` you write, to do something outside the traced computation
  each step (print a metric, write a checkpoint). They don't affect
  `weave.loop`'s control flow at all. `Callback.print` supports `every=`
  to throttle how often it actually prints, via `jax.lax.cond`.

- **Buffered logging** (`Callback.log`): a factory -- unlike the
  side-effects above, it needs state that survives across steps (a
  batch of pending values), and `jax.lax.scan`/`fori_loop` are purely
  functional, so that state can't hide anywhere; it has to be threaded
  through your own `carry` explicitly. `Callback.log(path, every, ...)`
  returns `(log_fn, init_buffer)`: put `init_buffer` in your carry,
  call `log_fn(buffer, step, **values)` every step to get the next
  buffer, and every `every` steps it flushes the accumulated batch to
  disk in one `io_callback` instead of one per step.

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
import json
import os
import jax
import jax.numpy as jnp


class Callback:
    """Namespace for step side-effects and loop stop-conditions."""

    @staticmethod
    def print(step, fmt=None, every=1, **values):
        """
        Print step metrics from inside a traced `body_fn`, via
        `jax.debug.print` (safe to call under `jit`/`scan`) -- feels
        like Python's `print`, but the format string and its values
        stay separate (like `jax.debug.print`/`str.format`), since
        `values` are traced: an f-string would try to render them
        *before* `Callback.print` ever saw them, which fails (or
        prints a tracer repr) instead of the runtime value.

        Args:
            step: The current step (usually the scan/fori index).
            fmt: Optional format string, e.g. `"loss={loss}"` (plain
                `str.format`-style placeholders -- not an f-string).
                If omitted, defaults to printing every name in
                `values` as `name=value`.
            every: Only print on step indices where `step % every == 0`
                (default `1`: print every call). Implemented with
                `jax.lax.cond`, so a skipped step costs only a scalar
                comparison -- `jax.debug.print` itself is never reached
                on a skipped step.
            **values: Named values available to `fmt` (or printed
                as-is if `fmt` is omitted), e.g. `loss=loss, lr=lr`.

        Example:
            >>> def body_fn(carry, i):
            ...     ...
            ...     Callback.print(i, "loss={loss} lr={lr}", every=10,
            ...                    loss=loss, lr=lr)
            ...     return new_carry, loss
        """
        if fmt is None:
            fmt = " ".join(f"{name}={{{name}}}" for name in values)
        template = "step={step} " + fmt

        def _do_print(_):
            jax.debug.print(template, step=step, **values)
            return jnp.zeros((), dtype=jnp.bool_)

        def _skip(_):
            return jnp.zeros((), dtype=jnp.bool_)

        if every > 1:
            should_print = (jnp.asarray(step) % every) == 0
            jax.lax.cond(should_print, _do_print, _skip, None)
        else:
            _do_print(None)

    @staticmethod
    def log(path, every=50, name="log", **fields):
        """
        Build a buffered, batched logger: `(log_fn, init_buffer)`.
        Meant for exactly what per-step file I/O is too slow for --
        instead of writing to disk every step, values are accumulated
        into an in-carry buffer of `every` steps and flushed to a
        `.jsonl` file (one JSON object per step) in `path` only once
        that buffer fills, via `jax.lax.cond` + `jax.experimental.
        io_callback` (the same pattern as `checkpointer`).

        Because `jax.lax.scan`/`fori_loop` are purely functional, the
        buffer can't live as hidden state anywhere -- it has to be
        threaded through `carry` explicitly, like any other value
        `body_fn` updates and returns. `log_fn` takes the current
        buffer and returns the next one; you're responsible for
        putting it in your carry tuple and getting it back out.

        Args:
            path: Directory to write the log file into -- same
                directory you'd pass to `checkpointer`/`load_struct`
                for this training run, so a run's checkpoint and its
                logs live side by side.
            every: Buffer size / flush period -- values are collected
                for `every` steps, then written to disk as one batch
                and the buffer resets. Default `50`.
            name: Log filename stem (default `"log"`); written as
                `{path}/{name}.jsonl`, appended to on every flush.
            **fields: Field name -> dtype (e.g. `loss=jnp.float32,
                lr=jnp.float32`), declaring the buffer's fixed shape.
                Every step must log exactly these fields.

        Returns:
            `(log_fn, init_buffer)`:
              - `init_buffer`: the buffer's initial (empty, zeroed)
                value -- put this in your initial carry.
              - `log_fn(buffer, step, **values) -> new_buffer`: call
                once per step with the current buffer and this step's
                values (matching `**fields` by name); returns the
                buffer to carry into the next step.

        Example:
            >>> def setup(self):
            ...     self.log_fn, self.log_buffer0 = Callback.log(
            ...         self.path, every=50, loss=jnp.float32, lr=jnp.float32,
            ...     )
            ...
            >>> def step(self, carry, i):
            ...     model, opt_state, log_buffer = carry
            ...     ...
            ...     log_buffer = self.log_fn(log_buffer, i, loss=loss, lr=lr)
            ...     return (model, opt_state, log_buffer), loss
            ...
            >>> def run(self):
            ...     init = (self.model, self.opt_state, self.log_buffer0)
            ...     (final_model, final_opt_state, _), losses = loop(
            ...         self.step, init, steps=1000,
            ...     )
        """
        field_names = list(fields.keys())
        init_buffer = {
            "step": jnp.zeros((every,), dtype=jnp.int32),
            "values": {
                field: jnp.zeros((every,), dtype=dtype)
                for field, dtype in fields.items()
            },
            "count": jnp.zeros((), dtype=jnp.int32),
        }

        def _flush(step_arr, values_dict, count):
            os.makedirs(path, exist_ok=True)
            n = int(count)
            out_path = os.path.join(path, f"{name}.jsonl")
            with open(out_path, "a") as f:
                for i in range(n):
                    record = {"step": int(step_arr[i])}
                    for field in field_names:
                        record[field] = float(values_dict[field][i])
                    f.write(json.dumps(record) + "\n")

        def log_fn(buffer, step, **values):
            missing = set(field_names) - set(values.keys())
            extra = set(values.keys()) - set(field_names)
            if missing or extra:
                raise ValueError(
                    f"Callback.log: expected fields {field_names}, got {list(values.keys())}"
                )

            idx = buffer["count"]
            new_step = buffer["step"].at[idx].set(jnp.asarray(step, dtype=jnp.int32))
            new_values = {
                field: buffer["values"][field].at[idx].set(jnp.asarray(values[field]))
                for field in field_names
            }
            new_count = idx + 1

            full_buffer = {"step": new_step, "values": new_values, "count": new_count}

            def _do_flush(_):
                jax.experimental.io_callback(
                    _flush, None, new_step, new_values, new_count, ordered=True,
                )
                return {
                    "step": jnp.zeros((every,), dtype=jnp.int32),
                    "values": {
                        field: jnp.zeros((every,), dtype=fields[field])
                        for field in field_names
                    },
                    "count": jnp.zeros((), dtype=jnp.int32),
                }

            def _keep(_):
                return full_buffer

            is_full = new_count >= every
            return jax.lax.cond(is_full, _do_flush, _keep, None)

        return log_fn, init_buffer

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
