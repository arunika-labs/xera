"""
Shard module: automatic device-sharding decorator for JAX functions.

Everything else in `xera.weave` assumes a single logical device: the
training loop, the optimizers, the callbacks. `shard` is the one piece
concerned with *where* an array's data actually lives across multiple
devices -- it exists so a function can declare its sharding intent
inline, without the caller having to construct a `Mesh` by hand
somewhere else in the script.

`shard` wraps `jax.device_put` + `jax.sharding.NamedSharding`. Given a
`jax.sharding.PartitionSpec` per argument, it:

- builds a device mesh automatically from `jax.devices()`, inferring
  axis names straight from the specs you pass in -- no `Mesh(...)` call
  needed anywhere in user code;
- shards each argument accordingly right before the wrapped function
  runs, so it composes naturally with `jax.jit`;
- falls back to a no-op (with a one-time warning, not an error) when
  only a single device is visible, since sharding a single device is
  meaningless and scripts should still run unmodified on a laptop;
- raises a clear `ValueError` -- not a raw `jax` traceback -- when more
  than one device *is* available but an argument's shape can't be
  evenly divided the way its spec asks for, since that almost always
  means a real mismatch between the data and the requested layout.

Example:
    >>> import jax
    >>> from jax.sharding import PartitionSpec as P
    >>> from xera.weave import shard
    >>>
    >>> @jax.jit
    ... @shard(P('data', None), P(None, 'model'))
    ... def forward(x, w):
    ...     return x @ w

Arguments without a matching spec (positional specs shorter than the
call's argument list, or keyword names absent from `**kwspecs`) are
passed through untouched -- `shard` only touches what you tell it to.
"""

from __future__ import annotations

import functools
import warnings

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding


_default_mesh_cache = {}  # axis_names (tuple) -> Mesh
_warned_single_device = False


def _collect_axis_names(specs, kwspecs):
    """Collect all unique axis names used across the given PartitionSpecs,
    preserving first-seen order."""
    axis_names = []
    for spec in list(specs) + list(kwspecs.values()):
        if spec is None:
            continue
        for ax in spec:
            if ax is not None and ax not in axis_names:
                axis_names.append(ax)
    return tuple(axis_names)


def _factorize_prioritize_first(n, num_axes):
    """Find a combination of `num_axes` factors whose product equals
    `n`, searched so that earlier axes get priority on larger factors.
    If no clean combination exists, the first axis takes every device
    and the remaining axes default to 1 (always valid for any `n`)."""

    def _find_factors(n, k):
        if k == 1:
            return [n]
        for f in range(n, 0, -1):
            if n % f == 0:
                rest = _find_factors(n // f, k - 1)
                if rest is not None:
                    return [f] + rest
        return None

    factors = _find_factors(n, num_axes)
    if factors is not None:
        return tuple(factors)

    return tuple([n] + [1] * (num_axes - 1))


def _build_mesh(axis_names, devices):
    """Build a mesh over `devices` for the given `axis_names`, using a
    first-axis-greedy factorization of the device count."""
    n = len(devices)
    num_axes = len(axis_names)

    if num_axes == 0:
        return None

    shape = (n,) if num_axes == 1 else _factorize_prioritize_first(n, num_axes)

    if int(np.prod(shape)) != n:
        raise ValueError(
            f"[xera.weave.shard] Failed to build mesh: device count "
            f"({n}) cannot be evenly reshaped into {shape} for axes "
            f"{axis_names}."
        )

    devices_arr = np.array(devices).reshape(shape)
    return Mesh(devices_arr, axis_names=axis_names)


def _get_or_build_mesh(axis_names):
    if axis_names in _default_mesh_cache:
        return _default_mesh_cache[axis_names]
    mesh = _build_mesh(axis_names, jax.devices())
    _default_mesh_cache[axis_names] = mesh
    return mesh


def _safe_device_put(array, mesh, spec, fn_name, arg_label):
    """Wrap `jax.device_put` so a shape/spec mismatch raises a clear,
    module-consistent `ValueError` instead of a raw internal traceback."""
    try:
        return jax.device_put(array, NamedSharding(mesh, spec))
    except Exception as e:  # noqa: BLE001 - intentionally broad, re-raised clearly
        raise ValueError(
            f"[xera.weave.shard] Failed to shard {arg_label} in "
            f"function '{fn_name}': shape {getattr(array, 'shape', '?')} "
            f"does not evenly divide according to spec {spec} on mesh "
            f"{mesh}. Make sure every sharded dimension size is evenly "
            f"divisible by the device count on its corresponding axis.\n"
            f"Original error: {e}"
        ) from e


def shard(*specs, **kwspecs):
    """
    Decorator: shard positional/keyword arguments across devices before
    calling the wrapped function.

    Args:
        *specs: `PartitionSpec` for positional arguments, in the same
            order as the wrapped function's parameters. May be shorter
            than the actual argument list -- trailing arguments are
            left un-sharded. Pass `None` for an argument you want left
            alone (equivalent to omitting it).
        **kwspecs: `PartitionSpec` for keyword arguments, keyed by
            parameter name.

    Returns:
        A decorator that, when applied to `fn`, shards matching
        arguments via `jax.device_put` + `NamedSharding` before calling
        `fn`, using a mesh built automatically from `jax.devices()`.

    Behavior:
        - Single device visible: sharding is skipped entirely and `fn`
          runs unmodified; a warning fires once per process, not once
          per call.
        - Multiple devices visible but a spec's axis sizes don't evenly
          divide the device count, or an argument's shape doesn't
          evenly divide across its spec: raises `ValueError`.

    Example:
        >>> @jax.jit
        ... @shard(P('data', None), P(None, 'model'))
        ... def forward(x, w):
        ...     return x @ w
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            global _warned_single_device

            devices = jax.devices()
            n_devices = len(devices)

            if n_devices <= 1:
                if not _warned_single_device:
                    warnings.warn(
                        f"[xera.weave.shard] Only {n_devices} device "
                        f"detected ({devices}). Sharding is skipped, "
                        f"function '{fn.__name__}' runs without "
                        f"sharding.",
                        stacklevel=2,
                    )
                    _warned_single_device = True
                return fn(*args, **kwargs)

            axis_names = _collect_axis_names(specs, kwspecs)
            if not axis_names:
                return fn(*args, **kwargs)

            mesh = _get_or_build_mesh(axis_names)

            new_args = []
            for i, a in enumerate(args):
                if i < len(specs) and specs[i] is not None:
                    a = _safe_device_put(a, mesh, specs[i], fn.__name__, arg_label=f"argument #{i}")
                new_args.append(a)

            new_kwargs = {}
            for k, v in kwargs.items():
                if k in kwspecs and kwspecs[k] is not None:
                    v = _safe_device_put(v, mesh, kwspecs[k], fn.__name__, arg_label=f"argument '{k}'")
                new_kwargs[k] = v

            return fn(*new_args, **new_kwargs)

        return wrapper

    return decorator


__all__ = ["shard"]
