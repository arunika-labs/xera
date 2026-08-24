"""
`.sxera` checkpoint format: model + optimizer + metadata in one file.

A `.sxera` file is a single, ordinary safetensors file -- it can be
opened with any standard safetensors reader. Xera just uses three key
prefixes to keep the three components apart:

    model.<path>       leaves of the model (a `Module`), e.g. "model.weight"
    optimizer.<path>    leaves of the optimizer (state), e.g. "optimizer.m"
    meta.<path>          array-valued leaves of `metadata` (e.g. an RNG key)

Anything in `metadata` that isn't a JAX array (plain Python config: step
counters, hyperparameters, strings, ...) is not a tensor at all -- it's
JSON-encoded and stamped into the safetensors metadata header, alongside
a `repr()` of each of the three pytrees' `treedef` (mirroring
`xera.serialize.state`'s drift-detection stamp). This is exactly the
*template* pattern used by `save_model`/`save_state`: the file holds
only leaves, never a pickled tree, and loading requires a `template`
with the same architecture/config for each of the three parts.

Use `save_struct`/`load_struct` for full round-tripping (typically via
`Struct.save_struct`), and `extract_model` when you only need the model
back out of a `.sxera` file as a plain `model.safetensors` (no template
required, since `.sxera` keys are already fully-qualified paths).
"""

from __future__ import annotations
import json
import jax
import numpy as np
from safetensors.numpy import save_file, load_file
from safetensors import safe_open
from .model import _key

_MODEL_PREFIX = "model."
_OPTIMIZER_PREFIX = "optimizer."
_META_PREFIX = "meta."

_META_MODEL_TREEDEF = "xera_sxera_model_treedef"
_META_OPTIMIZER_TREEDEF = "xera_sxera_optimizer_treedef"
_META_METADATA_TREEDEF = "xera_sxera_metadata_treedef"
_META_STATIC_JSON = "xera_sxera_static_json"


def _is_array_leaf(leaf):
    """True for anything that should round-trip as a safetensors tensor
    (has both a shape and a dtype) -- JAX/NumPy arrays, not plain Python
    scalars, strings, or None."""
    return hasattr(leaf, "shape") and hasattr(leaf, "dtype")


def _flatten_prefixed(pytree, prefix):
    """
    Flatten a pytree to {prefix+path: np.ndarray} plus its treedef.

    Only array-like leaves (things with `.shape`/`.dtype`) become
    tensors. Non-array leaves (plain Python ints, strings, `None`, ...)
    are skipped here -- they still round-trip via the treedef itself
    (JAX pytree leaves are part of the flatten/unflatten contract
    regardless of type) and, for `metadata`, are additionally captured
    by `_extract_static`'s JSON snapshot.
    """
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(pytree)
    tensors = {
        prefix + _key(p): np.asarray(leaf)
        for p, leaf in leaves_with_path
        if _is_array_leaf(leaf)
    }
    return tensors, treedef


def _unflatten_prefixed(template, prefix, tensors, release):
    """
    Rebuild a pytree from `template`'s structure using leaves read out of
    `tensors` (a flat {key: np.ndarray} dict already loaded from disk).

    Non-array leaves in `template` (plain Python ints, strings, `None`,
    ...) were never written as tensors -- `template`'s own value for
    those is kept as-is, regardless of `release`, since there is nothing
    on disk to compare or load for them.

    If `release=True` and an *array* leaf's key is missing from `tensors`
    (e.g. the template gained a new field since the checkpoint was
    written), the template's own leaf value is kept instead of raising.
    """
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(template)
    leaves = []
    for p, leaf in leaves_with_path:
        if not _is_array_leaf(leaf):
            leaves.append(leaf)
            continue
        key = prefix + _key(p)
        if key in tensors:
            leaves.append(
                jax.numpy.asarray(tensors[key]).reshape(leaf.shape).astype(leaf.dtype)
            )
        elif release:
            leaves.append(leaf)
        else:
            raise ValueError(
                f"load_struct: key '{key}' not found in checkpoint and "
                "release=False. If the structure intentionally changed, "
                "pass release=True."
            )
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _check_drift(name, saved_repr, template_treedef, release):
    if release or saved_repr is None:
        return
    if saved_repr != repr(template_treedef):
        raise ValueError(
            f"load_struct: {name}'s structure/config doesn't match the "
            f"checkpoint.\n  saved:    {saved_repr}\n"
            f"  template: {template_treedef!r}\n"
            "If this change is intentional, pass release=True."
        )


def save_struct(model, optimizer, metadata, path):
    """
    Save a model, an optimizer (state), and metadata together as one
    `.sxera` file.

    Args:
        model: The model (`Module` instance) to save.
        optimizer: The optimizer (state) to save.
        metadata: A dict or `Struct` holding everything else needed to
            resume training (step counters, RNG keys, config, ...). JAX
            array leaves are stored as tensors (prefixed `meta.`); every
            other (static, non-array) value is JSON-encoded into the
            file's metadata header.
        path: Destination file path (conventionally ending in `.sxera`).

    Example:
        >>> save_struct(
        ...     model, optimizer,
        ...     metadata={"step": 1000, "key": rng_key},
        ...     path="ckpt.sxera",
        ... )
    """
    model_tensors, model_treedef = _flatten_prefixed(model, _MODEL_PREFIX)
    opt_tensors, opt_treedef = _flatten_prefixed(optimizer, _OPTIMIZER_PREFIX)
    meta_tensors, meta_treedef = _flatten_prefixed(metadata, _META_PREFIX)

    tensors = {**model_tensors, **opt_tensors, **meta_tensors}

    # Anything in `metadata` that isn't a JAX-array leaf (plain config:
    # ints, strings, dicts of hyperparameters, ...) is static from JAX's
    # point of view and lives only in the treedef -- JSON-encode it so
    # it survives on disk too, not just as an opaque repr().
    static_json = json.dumps(_extract_static(metadata))

    header = {
        _META_MODEL_TREEDEF: repr(model_treedef),
        _META_OPTIMIZER_TREEDEF: repr(opt_treedef),
        _META_METADATA_TREEDEF: repr(meta_treedef),
        _META_STATIC_JSON: static_json,
    }
    save_file(tensors, path, metadata=header)


def _extract_static(metadata):
    """
    Best-effort JSON-safe snapshot of `metadata`'s non-array values.

    Plain `dict` metadata is walked directly. `Struct`/other pytree
    metadata is not walked field-by-field here (its static config is
    already captured in `meta_treedef`'s repr, and reconstructing it
    lives entirely with `template` at load time) -- this only produces
    a convenience JSON snapshot for external inspection of the file.
    """
    if isinstance(metadata, dict):
        def _safe(v):
            try:
                json.dumps(v)
                return v
            except TypeError:
                return repr(v)
        return {k: _safe(v) for k, v in metadata.items() if not hasattr(v, "shape")}
    return {}


def _apply_static_json(template, static_json):
    """
    Overlay JSON-decoded static values from the header onto `template`'s
    own non-array leaves (dict metadata only -- see `_extract_static`).
    Array leaves are untouched here; those come from tensors instead.
    """
    if not isinstance(template, dict) or not static_json:
        return template
    try:
        static = json.loads(static_json)
    except (TypeError, json.JSONDecodeError):
        return template
    merged = dict(template)
    for k, v in static.items():
        if k in merged and not _is_array_leaf(merged[k]):
            merged[k] = v
    return merged


def load_struct(model_template, optimizer_template, metadata_template, path, release=False):
    """
    Load a `.sxera` checkpoint back into a model, optimizer, and metadata.

    Args:
        model_template: A model with the same architecture as the saved
            model (e.g. a freshly constructed, uninitialized instance).
        optimizer_template: An optimizer (state) with the same structure
            as the saved optimizer.
        metadata_template: A dict/`Struct` with the same structure as the
            saved metadata. For plain-`dict` metadata, non-array values
            (step counters, plain config, ...) are overlaid from the
            checkpoint's JSON snapshot regardless of `release`, since a
            changed *value* for an existing key is not a structural
            drift -- only a changed *shape* (new/removed keys) is.
        path: The `.sxera` file to load.
        release: If `False` (default), any structural/config mismatch
            between a template and what was stamped at save time raises
            `ValueError` for that component. If `True`, mismatches are
            treated as intentional: each template's own structure/config
            is used as-is, missing leaves fall back to the template's
            values, and the next checkpoint written from the resumed run
            will reflect the new structure.

    Returns:
        A tuple `(model, optimizer, metadata)`.

    Example:
        >>> model, optimizer, meta = load_struct(
        ...     MyModel(), Adam(lr=1e-3).init(MyModel()), {"step": 0, "key": None},
        ...     "ckpt.sxera",
        ... )
        >>> # Optimizer hyperparameters changed since this checkpoint was saved:
        >>> model, optimizer, meta = load_struct(
        ...     MyModel(), Adam(lr=5e-4).init(MyModel()), {"step": 0, "key": None},
        ...     "ckpt.sxera", release=True,
        ... )
    """
    with safe_open(path, framework="numpy") as f:
        header = f.metadata() or {}
    tensors = load_file(path)

    _, model_treedef = jax.tree_util.tree_flatten_with_path(model_template)
    _, opt_treedef = jax.tree_util.tree_flatten_with_path(optimizer_template)
    _, meta_treedef = jax.tree_util.tree_flatten_with_path(metadata_template)

    _check_drift("model", header.get(_META_MODEL_TREEDEF), model_treedef, release)
    _check_drift("optimizer", header.get(_META_OPTIMIZER_TREEDEF), opt_treedef, release)
    _check_drift("metadata", header.get(_META_METADATA_TREEDEF), meta_treedef, release)

    model = _unflatten_prefixed(model_template, _MODEL_PREFIX, tensors, release)
    optimizer = _unflatten_prefixed(optimizer_template, _OPTIMIZER_PREFIX, tensors, release)
    metadata_template = _apply_static_json(metadata_template, header.get(_META_STATIC_JSON))
    metadata = _unflatten_prefixed(metadata_template, _META_PREFIX, tensors, release)

    return model, optimizer, metadata


def extract_model(sxera_path, out_path):
    """
    Pull just the model out of a `.sxera` checkpoint into a plain
    `model.safetensors` file -- no template required, since `.sxera`
    tensor keys are already fully-qualified paths (just re-keyed without
    the `model.` prefix).

    Args:
        sxera_path: Path to the source `.sxera` file.
        out_path: Destination path for the extracted model safetensors file.

    Example:
        >>> extract_model("ckpt.sxera", "model.safetensors")
        >>> # Now loadable the ordinary way:
        >>> model = load_model(MyModel(), "model.safetensors")
    """
    tensors = load_file(sxera_path)
    model_tensors = {
        key[len(_MODEL_PREFIX):]: value
        for key, value in tensors.items()
        if key.startswith(_MODEL_PREFIX)
    }
    save_file(model_tensors, out_path)


__all__ = ["save_struct", "load_struct", "extract_model"]
