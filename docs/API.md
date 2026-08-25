# xera — API Guide

This guide covers the full public API of xera: the model-building side
(`xera.loom` for layers, `xera.loom.functional` for activations and
attention) and the training side (`xera.weave` for the training loop,
losses, optimizers, callbacks, and sharding, plus `xera.serialize` for
checkpointing), all sitting on top of the shared `xera.core` base
classes (`Module` for model parameters, `Struct` for everything that
drives or observes training).

Conventional aliases used throughout this guide and recommended for
your own code:

```python
import xera.loom as L               # layers/modules
import xera.loom.functional as F    # activations + functional ops
import xera.weave as W              # training loop, losses, callbacks, sharding
import xera.weave.optimizer as O    # optimizers
import xera.serialize as S          # model and checkpoint save/load
```

Everything on the model side is a `Module`: a dataclass-like,
JAX-pytree layer with parameters, composed by nesting (a
`TransformerBlock` holds a `MultiHeadAttention` and an `MLP`, an `MLP`
holds two `Dense`, and so on). Everything on the training side is a
`Struct`: the same dataclass/pytree mechanics, but for things that
drive or observe training rather than differentiable parameters —
datasets, optimizers, train steps, and the top-level `Trainer` itself.

## Contents

**Part I — `xera.loom`: building models**

1. [`Module` — the base every layer is built on](#1-module--the-base-every-layer-is-built-on)
2. [`L.Dense` — the linear layer](#2-ldense--the-linear-layer)
3. [Convolution and pooling](#3-convolution-and-pooling)
4. [Embeddings](#4-embeddings)
5. [Normalization](#5-normalization)
6. [`L.Dropout`](#6-ldropout)
7. [Attention](#7-attention)
8. [Transformer blocks](#8-transformer-blocks)
9. [Recurrent / state-space: `SSM`, `SelectiveSSM`, `MambaBlock`](#9-recurrent--state-space-ssm-selectivessm-mambablock)
10. [Combinators: `Sequential`, `Residual`, `Lambda`](#10-combinators-sequential-residual-lambda)
11. [`F` — functional ops](#11-f--functional-ops)
12. [Sharding a model's parameters (`W.shard`, model-parallel)](#12-sharding-a-models-parameters-wshard-model-parallel)

**Part II — `xera.weave`: training models**

13. [`Struct` — the base of everything on the training side](#13-struct--the-base-of-everything-on-the-training-side)
14. [`W.loop` — running the training loop](#14-wloop--running-the-training-loop)
15. [`W.Loss` — loss functions](#15-wloss--loss-functions)
16. [`O` — optimizers](#16-o--optimizers)
17. [`W.shard` — device sharding](#17-wshard--device-sharding)
18. [`S` — checkpointing](#18-s--checkpointing)
19. [`W.Callback` — side-effects and stop conditions](#19-wcallback--side-effects-and-stop-conditions)
20. [Full example](#20-full-example)

---

# Part I — `xera.loom`: building models

Nothing in this part needs `jax.jit`/`vmap`/`grad` applied manually —
layers are just called like functions (`layer(x)`) and compose with
JAX transforms the ordinary way from the outside.

---

## 1. `Module` — the base every layer is built on

`xera.core.Module` (re-exported as `xera.loom.Module`, alongside
`Buffer`, `RNGPool`, and `param`) is what `Dense`, `Conv`,
`MultiHeadAttention`, and every other layer in this guide subclasses.
You'll want it directly as soon as you write a custom layer.

```python
import xera.loom as L
from xera.loom import param #Optional
import xera.initializers as init

class MyLinear(L.Module):
    in_features: int
    out_features: int

    def setup(self):
        self.weight = param(self.rng(), init.xavier_normal(), (self.in_features, self.out_features))
        self.bias = param(self.rng(), init.zeros(), (self.out_features,))

    def __call__(self, x):
        return x @ self.weight + self.bias

layer = MyLinear(10, 20, key=jax.random.PRNGKey(0))
output = layer(jnp.ones((5, 10)))   # (5, 20)
```

- **Fields are declared like a dataclass** (`name: type` or
  `name: type = default`), set positionally or by keyword at
  construction: `MyLinear(10, 20, key=...)` or
  `MyLinear(in_features=10, out_features=20, key=...)`.
- **`setup()`** runs once, immediately after fields are assigned, and
  is where you create parameters (and any sub-layers). It's a no-op by
  default, so a layer with nothing to initialize (`Dropout`, `Lambda`,
  the combinators in §10) simply omits it.
- **`key=`** is required for any layer whose `setup()` calls
  `self.rng()` — omitting it raises `RuntimeError` at the first
  `self.rng()` call, not silently. Layers that hold sub-layers pass
  `key=self.rng()` down to each child so every parameter in the tree
  traces back to one root key deterministically:

  ```python
  self.q_proj = Dense(self.dim, self.dim, key=self.rng())
  self.k_proj = Dense(self.dim, self.dim, key=self.rng())
  ```
- **`self.rng(n=None)`** returns one fresh key (`n=None`, the default)
  or a list of `n` independent keys — see `SSM`'s `setup()` (§9) for
  the multi-key form: `k_B, k_C, k_dt = self.rng(3)`.
- **`param(key, init_fn, shape, dtype=jnp.float32)`** is the plain
  helper every layer's `setup()` calls to materialize a parameter:
  `init_fn(key, shape, dtype)`. Initialization functions live in
  `xera.initializers` — `zeros()`, `ones()`, `normal(stddev=0.05)`,
  `xavier_normal()`, `xavier_uniform()`, `kaiming_normal()`,
  `kaiming_uniform()`, `lecun_normal()`, `constant(value=0.0)`,
  `truncated_normal(stddev=0.05)`, `orthogonal(scale=1.0)`,
  `variance_scaling(scale=1.0, mode="fan_in", distribution="normal")`
  — each a factory returning an `(key, shape, dtype) -> array`
  function, not the array itself.
- **`Buffer`** wraps a value that should live in the pytree as a leaf
  but isn't a trainable parameter — running statistics being the one
  case in this guide (`BatchNorm`/`GroupNormWithRunningStats`, §5).
  Because `Module` instances aren't mutated in place under JAX
  transforms, updating a buffer means producing a *new* layer instance
  with the buffer replaced — see §5 for the pattern.
- Two Modules with the same fields are still different pytree
  *treedefs* if their non-array (static) fields differ, same as
  `Struct` in the training guide — this is what lets `S.load_struct`
  (§18.3) detect an accidentally-changed architecture.

---

## 2. `L.Dense` — the linear layer

```python
layer = L.Dense(in_features=128, out_features=64, key=jax.random.PRNGKey(0))
y = layer(x)   # x: (..., 128) -> y: (..., 64)
```

`y = x @ weight + bias`. `weight` is `(in_features, out_features)`,
initialized with `lecun_normal()`; `bias` is `(out_features,)`,
initialized to zero, and present unless `use_bias=False`. This is the
layer nearly everything else in this guide (`MultiHeadAttention`'s
projections, `MLP`, `SelectiveSSM`'s input projections, ...) is built
out of.

---

## 3. Convolution and pooling

### 3.1 `L.Conv` / `L.ConvTranspose`

```python
conv = L.Conv(in_channels=3, out_channels=64, kernel_size=(3, 3), key=key)
y = conv(x)   # x: (batch, *spatial, 3) -> y: (batch, *spatial', 64)
```

Both expect **channels-last** input: `(batch, *spatial, channels)` —
one spatial dim for 1D conv, two for images, etc., inferred from
`len(kernel_size)`. Shared options: `stride` (default `1`), `padding`
(`"SAME"`, `"VALID"`, or an explicit tuple of `(low, high)` pairs per
spatial dim — used this way for causal 1D conv in `MambaBlock`, §9),
`dilation` (default `1`), `use_bias` (default `True`). `Conv` also
takes `groups` (default `1`) for grouped/depthwise convolution — every
channel its own group, as `MambaBlock` uses it for a depthwise causal
conv. Weights are `kaiming_normal()`-initialized.

### 3.2 `L.MaxPool` / `L.AvgPool` / `L.GlobalAvgPool`

```python
pool = L.MaxPool(pool_size=(2, 2), stride=(2, 2), padding="VALID")
y = pool(x)   # halves spatial dims
```

`MaxPool`/`AvgPool` take `pool_size`, optional `stride` (defaults to
`pool_size` — non-overlapping windows), and `padding` (`"VALID"` or
`"SAME"`). `GlobalAvgPool(keepdims=False)` averages over every spatial
dimension in one call — `(batch, *spatial, channels)` ->
`(batch, channels)` (or with the spatial dims kept as size-1 if
`keepdims=True`). None of the three take a `key` (no parameters).

---

## 4. Embeddings

### 4.1 `L.Embedding`

```python
embed = L.Embedding(num_embeddings=10000, features=256, key=key)
vecs = embed(token_ids)   # (...,) int indices -> (..., 256)
```

A `(num_embeddings, features)` lookup table (`normal(stddev=0.02)`
init) indexed as `weight[idx]` — `idx` can be any shape, including a
trailing `seq_len`.

### 4.2 `L.RotaryEmbedding`

```python
rope = L.RotaryEmbedding(dim=64, base=10000.0)   # no key -- no parameters
q_rot = rope(q)                # x: (..., seq_len, dim)
k_rot = rope(k, offset=past_len)  # offset shifts the position index, for KV-cache decoding
```

RoPE has no learned parameters (no `key=` needed at construction) —
it's a fixed rotation computed from position and `dim`/`base`, applied
to query/key vectors before the attention dot product. `offset` shifts
where position `0` starts, for incrementally decoding past a cached
prefix. `MultiHeadAttention`/`GroupedQueryAttention` construct one of
these internally when `use_rope=True` (§7).

---

## 5. Normalization

### 5.1 Stateless: `LayerNorm`, `RMSNorm`, `GroupNorm`, `InstanceNorm`, `LayerScale`

```python
ln = L.LayerNorm(dim=512, key=key)
y = ln(x)   # x: (..., 512)
```

All five just take `x` and return a normalized tensor of the same
shape — no `deterministic`/train-vs-eval distinction, since none of
them carry running statistics:

- **`LayerNorm(dim, eps=1e-5)`** — normalizes over the last axis
  (mean + variance), learned `gamma`/`beta` (init `ones`/`zeros`).
- **`RMSNorm(dim, eps=1e-6)`** — `LayerNorm` without mean-centering,
  RMS-scaling only, learned `gamma` (no `beta`). Cheaper, common in
  LLMs.
- **`GroupNorm(num_groups, dim, eps=1e-5)`** — splits the channel dim
  into `num_groups` (must divide `dim`), normalizes within each group
  over `(spatial..., group_channels)`. Input `(batch, *spatial, dim)`.
- **`InstanceNorm(dim, eps=1e-5)`** — normalizes per-sample over all
  spatial dims (no batch mixing, no grouping). Input
  `(batch, *spatial, dim)`.
- **`LayerScale(dim, init_value=1e-5)`** — a single learned per-channel
  scale (`x * scale`), no centering/variance at all — used to
  stabilize very deep residual stacks, not a normalizer in the
  statistical sense.

### 5.2 Stateful: `BatchNorm`, `GroupNormWithRunningStats`

These carry running statistics as `Buffer`s (`running_mean`,
`running_var`), so they need a training/eval mode and return an
**updated layer alongside the output**:

```python
bn = L.BatchNorm(dim=64, momentum=0.9, key=key)

y_train, bn = bn(x, deterministic=False)   # batch stats; running stats updated; rebind bn
y_eval, _   = bn(x, deterministic=True)    # running stats (default); layer unchanged
```

- **`deterministic=True`** (the default) uses the stored running
  stats and returns `(output, self)` — the exact same object, so no
  update actually happens.
- **`deterministic=False`** (training) computes batch statistics,
  blends them into the running stats by `momentum` (`new = momentum *
  old + (1 - momentum) * batch`), and returns `(output, new_layer)` —
  a genuinely new instance with the updated `Buffer`s. **You must
  rebind your `bn`/`gn` variable to this new instance** (as in the
  `y_train, bn = ...` line above) or the running-stat update is
  silently dropped on the next call — `Module`s are immutable data
  under JAX transforms, so "updating" one always means "producing a
  new one."
- `GroupNormWithRunningStats(num_groups, dim, momentum=0.9, eps=1e-5)`
  is the same pattern, group-normalized instead of batch-normalized.
- §10 (`Sequential`/`Residual`) shows how this "maybe returns a tuple"
  shape is handled automatically when these sit inside a composed
  model, so you don't have to special-case them yourself in every
  forward pass.

---

## 6. `L.Dropout`

```python
drop = L.Dropout(rate=0.1)   # no key needed at construction
y = drop(x, key=step_key, deterministic=False)   # training: drop + inverse-scale kept units
y = drop(x)                                       # eval (default): exact identity, x unchanged
```

- **`deterministic=True`** (the default, and also whenever `rate ==
  0.0`) is an **exact identity** — `x` is returned unchanged, no
  scaling applied in either direction.
- **`deterministic=False`** applies inverted dropout: each unit is
  independently zeroed with probability `rate` (needs `key=`), and
  every *kept* unit is scaled by `1 / (1 - rate)` so the expected
  value matches the un-dropped input — this is where the scaling
  happens, not at eval time.
- Every other layer in this guide that has a `dropout_rate`
  (`MultiHeadAttention`, `MLP`, `TransformerBlock`, ...) owns one of
  these internally and forwards its own `key`/`deterministic` straight
  through to it.

---

## 7. Attention

### 7.1 `causal_mask`

```python
mask = L.causal_mask(seq_len)   # (seq_len, seq_len) bool, True = allowed
```

A lower-triangular boolean mask — position `i` may attend to position
`j` only if `j <= i`. Pass it as `mask=` to any of the three attention
layers below.

### 7.2 `L.MultiHeadAttention`

```python
attn = L.MultiHeadAttention(dim=512, num_heads=8, dropout_rate=0.1, use_rope=False, key=key)
y = attn(x, mask=causal_mask(seq_len), key=dropout_key, deterministic=False)
```

Standard multi-head self-attention: separate `q_proj`/`k_proj`/
`v_proj`/`out_proj` `Dense` layers, `dim` split into `num_heads` of
`dim // num_heads` (must divide evenly), scaled dot-product scores
(`/ sqrt(head_dim)`), softmax, dropout on the attention weights (not
the output), then `out_proj`. `x`: `(batch, seq_len, dim)` in,
`(batch, seq_len, dim)` out. Set `use_rope=True` to rotate `q`/`k`
with `RotaryEmbedding(head_dim, rope_base)` (§4.2) before scoring —
built once in `setup()`, reused every call. `mask`/`key` are optional
keyword-only args; omit `key` (or pass `deterministic=True`, the
default) to skip dropout deterministically.

### 7.3 `L.GroupedQueryAttention`

```python
attn = L.GroupedQueryAttention(dim=512, num_heads=8, num_kv_heads=2, key=key)
y = attn(x)   # same shapes as MultiHeadAttention
```

Same interface and shapes as `MultiHeadAttention`, but `k_proj`/
`v_proj` project down to `num_kv_heads` (`num_heads` must divide
evenly by it) instead of `num_heads` — each KV head is shared by
`num_heads // num_kv_heads` query heads (repeated via `jnp.repeat`
before the dot product), trading a little quality for much smaller
KV-cache memory at inference. `num_kv_heads=1` is multi-query
attention; `num_kv_heads=num_heads` degenerates to ordinary MHA.

### 7.4 `L.SelfAttention`

```python
attn = L.SelfAttention(dim=256, dropout_rate=0.1, key=key)
y = attn(x)                          # self-attention
y = attn(x, context=encoder_out)     # cross-attention: q from x, k/v from context
```

The simple, single-head case — no head-splitting at all, one
`q_proj`/`k_proj`/`v_proj`/`out_proj`. Pass `context=` to use `x` only
for queries and a different tensor for keys/values (cross-attention,
e.g. decoder attending to an encoder's output); omitting `context`
makes it ordinary self-attention. `mask`/`key`/`deterministic` behave
the same as the other two.

---

## 8. Transformer blocks

### 8.1 `L.MLP`

```python
mlp = L.MLP(dim=512, hidden_dim=2048, dropout_rate=0.1, key=key)
y = mlp(x, key=dropout_key, deterministic=False)
```

The standard transformer feed-forward sublayer: `Dense(dim,
hidden_dim)` -> GELU -> `Dropout` -> `Dense(hidden_dim, dim)`. `x`:
`(..., dim)` in and out.

### 8.2 `L.TransformerBlock`

```python
block = L.TransformerBlock(dim=512, num_heads=8, mlp_hidden_dim=2048, dropout_rate=0.1, key=key)
y = block(x, mask=L.causal_mask(seq_len), key=step_key, deterministic=False)
```

Pre-norm transformer block: `x + attn(ln1(x))`, then `x + mlp(ln2(x))`
— `MultiHeadAttention` + `MLP` + two independent `LayerNorm`s, the
architecture behind GPT/BERT/ViT-style models. A single `key` (if
given) is split internally into separate attention-dropout and
MLP-dropout keys, so you only pass one key per block per step.

---

## 9. Recurrent / state-space: `SSM`, `SelectiveSSM`, `MambaBlock`

Linear-time alternatives to attention for long sequences, all built on
a diagonal state-space recurrence discretized with zero-order hold and
run via `jax.lax.scan` over the sequence dimension internally (you
don't drive the scan yourself).

```python
ssm = L.SSM(channels=64, state_dim=16, key=key)
y = ssm(x)   # x: (batch, seq_len, 64) -> y: (batch, seq_len, 64)
```

- **`SSM(channels, state_dim=16, dt_min=0.001, dt_max=0.1)`** — the
  S4D variant: a fixed (not input-dependent) diagonal state-space
  layer. `channels` in/out, `state_dim` is the hidden recurrence
  width.
- **`SelectiveSSM(d_inner, state_dim=16, dt_rank=None)`** — Mamba's
  core primitive: `dt`/`B`/`C` are computed *from the input* at every
  timestep (via a low-rank `x_proj`/`dt_proj`) instead of being fixed
  parameters, letting the model selectively remember/forget per
  position. `dt_rank` defaults to `max(1, d_inner // 16)`.
- **`MambaBlock(d_model, d_inner=None, state_dim=16, conv_kernel=4, dt_rank=None)`**
  — the full Mamba block: input projected to `2 * d_inner` and split
  into a gate `z` and a path `x_in`; `x_in` goes through a causal
  depthwise `Conv` (kernel `conv_kernel`, left-padded so position `i`
  never sees `i+1`) + SiLU, then `SelectiveSSM`, then gated by
  `silu(z)`, then projected back to `d_model`. `d_inner` defaults to
  `d_model * 2`.

```python
mamba = L.MambaBlock(d_model=512, key=key)
y = mamba(x)   # x: (batch, seq_len, 512) -> y: (batch, seq_len, 512)
```

---

## 10. Combinators: `Sequential`, `Residual`, `Lambda`

These exist so §5.2's stateful layers (`BatchNorm`,
`GroupNormWithRunningStats`) compose with everything else in this
guide **without every call site having to special-case them**.

```python
model = L.Sequential([
    L.Dense(784, 256, key=k1),
    L.Dense(256, 128, key=k2),
    L.Dense(128, 10, key=k3),
])
y = model(x)   # no stateful layers -> plain output, no tuple

model = L.Sequential([L.Dense(64, 64, key=k1), L.BatchNorm(dim=64, key=k2), L.Dropout(0.1)])
y, model = model(x, deterministic=False, key=step_key)   # has a stateful layer -> (output, new_model); rebind
y, _     = model(x, deterministic=True)                   # eval -> BatchNorm returns itself, still (output, self)
```

- **`Sequential(layers)`** calls each layer in order, forwarding
  `**kwargs` (`key`, `deterministic`, `mask`, ...) to each layer —
  **only the kwargs that layer's own `__call__` actually declares**
  (inspected via `inspect.signature`, so `Dropout` doesn't choke on an
  unexpected `mask` meant for an attention layer two steps later, and
  vice versa). If *any* layer in the list returns the
  `(output, new_layer)` stateful shape (§5.2) with a genuinely
  different `new_layer`, `Sequential` itself returns
  `(output, new_sequential)`; otherwise it returns just the output —
  you don't need to know in advance whether a given `Sequential`
  contains a stateful layer, only check the return type (or always
  unpack it as a tuple if you know it might, as above).
- **`Residual(inner)`** wraps one layer/sub-model as `x + inner(x)`,
  with the same automatic tuple-handling if `inner` is (or contains) a
  stateful layer.
- **`Lambda(fn)`** wraps a plain function with no parameters and no
  state — `Lambda(lambda x: jnp.maximum(0, x))` — for one-off
  transforms that don't need a full custom `Module`. `Lambda` ignores
  any `**kwargs` Sequential/Residual forward to it (`fn` is called as
  `fn(x)` only).

---

## 11. `F` — functional ops

`xera.loom.functional` mirrors `jax.nn`'s shape: mostly thin aliases,
plus one original implementation.

### 11.1 Activations — thin `jax.nn` aliases

```python
y = F.relu(x)
y = F.gelu(x)
y = F.softmax(x, axis=-1)
```

`celu`, `elu`, `gelu`, `glu`, `hard_sigmoid`, `hard_silu`,
`hard_swish`, `hard_tanh`, `leaky_relu`, `log_sigmoid`, `log_softmax`,
`logsumexp`, `mish`, `one_hot`, `relu`, `relu6`, `selu`, `sigmoid`,
`silu`, `soft_sign`, `softmax`, `softplus`, `squareplus`,
`standardize`, `swish`, `tanh` — every one is `jax.nn.<name>` itself,
re-exported so you don't need a separate `jax.nn` import; no
reimplementation, no behavior differences.

### 11.2 `F.auto_flash_attention` — the recommended attention entry point

```python
y = F.auto_flash_attention(q, k, v, causal=True)
```

`q`, `k`, `v`: `(batch, num_heads, seq_len, head_dim)`. Unlike the
layers in §7, this is a bare function (no `Dense` projections, no
parameters) — the scaled-dot-product-attention *primitive* underneath
a custom attention layer you write yourself, the same role
`jax.nn.dot_product_attention` plays in `jax.nn`.

Automatically picks a backend for the current device (`backend=None`,
the default):

| Platform | Backend | Requirements | Falls back to |
|---|---|---|---|
| TPU | Splash (Pallas kernel) | `bfloat16` only, no `bias`/`local_window_size` | `xenafl` |
| GPU | cuDNN fused attention | `bfloat16`/`float16` only (no `fp32`), no `bias`/`local_window_size`, Ampere+/sm_80+ | `xenafl` |
| anything else (CPU, ...) | `xenafl` | — (always works) | — |

When a vendor backend can't serve the request (wrong dtype, or `bias`/
`local_window_size` was requested), it silently drops to `xenafl` and
prints one line — `XeraInfo: AutoFA using 'xenafl', because <reason>.`
— not a `warnings.warn`, since this is routine/expected, not a
problem. Nothing is printed on CPU (there's no fallback happening,
`xenafl` is simply the only backend) and nothing is printed when the
vendor backend is used successfully.

Other args: `scale` (default `1/sqrt(head_dim)`), `bias` (additive,
only honored by `xenafl` — requesting it elsewhere routes there),
`local_window_size` (int, or `(left, right)` tuple; only `xenafl`).
Pass `backend="cudnn"`/`"splash"`/`"xenafl"` to force one explicitly —
forcing means "use exactly this or raise `ValueError`", never a
silent fallback.

### 11.3 `xenafl_attention` — the portable kernel, called directly

`auto_flash_attention` routes to this automatically whenever no vendor
backend applies; you'd reach for it directly only if you want manual
control over block/tile size, or want to guarantee the portable path
without going through dispatch logic at all.

```python
from xera.loom.flash_attention.xenafl_attention import xenafl_attention

y = xenafl_attention(
    q, k, v, bias,             # bias: array or None -- always positional, never omitted
    causal, scale, window_left, window_right,  # scale/window_*: value or None, always positional
    block_q, block_k,          # tile sizes along the query/key sequence dim
)
```

**Every argument is positional** — `q, k, v, bias, causal, scale,
window_left, window_right, block_q, block_k`, in that exact order,
none optional/keyword — a consequence of being wrapped in
`jax.custom_vjp` (needed for O(seq_len) memory on the backward pass
too, not just the forward; the module docstring in
`xera/loom/flash_attention/xenafl_attention.py` explains why
autodiff-through-scan alone isn't enough). Pass `None` explicitly for
`bias`/`scale`/`window_left`/`window_right` when you don't want them —
there's no default to fall back on the way there is with
`auto_flash_attention`. `block_q`/`block_k` are the fixed tile sizes
scanned over; `seq_len` need not be a multiple of either (the last
tile is padded and masked out internally).

O(seq_len) memory on both forward and backward by construction — the
full `(seq_len, seq_len)` score matrix is never materialized (block
tiling + online/running softmax), same algorithm regardless of which
platform runs it (pure `jnp`, no Pallas, no vendor kernel — that's
what makes it the universal fallback in §11.2).

---

## 12. Sharding a model's parameters (`W.shard`, model-parallel)

§17 below covers `xera.weave.shard` for **data-parallel**
sharding (splitting a batch across devices). The same decorator works
for **model-parallel** sharding — splitting a large layer's weights
themselves across devices — by decorating a small function whose
arguments are exactly the parameter arrays you want split, then
assigning the results back onto the layer:

```python
import jax
from jax.sharding import PartitionSpec as P
from xera.weave import shard

model = L.Dense(8192, 8192, key=jax.random.PRNGKey(0))

@shard(P(None, 'model'), P('model'))
def shard_dense_params(weight, bias):
    return weight, bias

model.weight, model.bias = shard_dense_params(model.weight, model.bias)
# model.weight columns and model.bias are now split across every visible
# device along a 'model' axis; model.weight's row dim stays whole on each.

y = model(x)   # forward works unchanged -- x need not be sharded for this to run
```

This works because a `Module` instance's attributes are ordinary
Python attributes (reassignable after construction, unlike a frozen
dataclass) that `Module`'s pytree flattening reads fresh every time —
so replacing `model.weight` with a sharded array is enough; nothing
else about the layer needs to change. Do this once, typically right
after constructing the model (or in a `Struct.setup()`, alongside
`S.load_struct`), not per-step — a parameter's sharding doesn't need
re-establishing every training step the way a fresh batch does.

Each parameter needs its own spec matched to its own shape — `weight`
is `(in_features, out_features)` here so `P(None, 'model')` shards its
*columns* (each device gets a full set of rows but only a slice of
output features); `bias` is `(out_features,)` so its matching spec is
just `P('model')`, one axis. There's no single spec that correctly
shards every parameter in a layer at once when shapes differ (a 2D
weight and a 1D bias can't share one `PartitionSpec`), which is why
this goes through a small dedicated function per layer (or per group
of same-shaped parameters) rather than one `@shard` call over the
whole model in one shot.

For a bigger model, apply the same pattern to whichever layer(s) are
actually large enough to need splitting (e.g. just an `MLP`'s two
`Dense`s, transformer-style):

```python
@shard(P(None, 'model'), P('model', None))
def shard_mlp(fc1_weight, fc2_weight):
    return fc1_weight, fc2_weight

mlp.fc1.weight, mlp.fc2.weight = shard_mlp(mlp.fc1.weight, mlp.fc2.weight)
```

`fc1`'s output dim and `fc2`'s input dim are the same `hidden_dim`
axis — sharding `fc1`'s columns and `fc2`'s rows the same way along
`'model'` keeps each device holding a matching slice of the hidden
dimension all the way through, without needing an all-gather in
between the two matmuls.

On a single visible device this all still runs unchanged (`shard`
skips sharding entirely, per §17); on multiple devices,
combine this with §5's data-parallel batch example if you want both a
sharded batch *and* sharded model weights in the same training step.

---

# Part II — `xera.weave`: training models

`xera.loom` (`L`, layers/modules) and `xera.loom.functional` (`F`,
functional ops) come up occasionally below wherever a model is needed,
but this part is about `weave`, not about building models — see Part I
above for that.

---

## 13. `Struct` — the base of everything on the training side

`xera.core.Struct` is not part of `weave` itself, but it's what you
build a `Trainer` out of, so start here.

`Struct` is the training-side counterpart to `Module` (which is for
model layers/parameters). Same mechanics — dataclass fields, JAX
pytree registration, an optional `setup()` hook — but meant for things
that drive or observe training rather than differentiable parameters:
datasets, optimizers, train steps, and the top-level training driver
itself.

```python
class Trainer(W.Struct):
    model: L.Dense = None
    optimizer: O.Adam = None

    def setup(self):
        ...           # runs once, right after fields are assigned

    def run(self):
        ...           # your training loop
```

Key behaviors:

- **Fields are declared like a dataclass** (`name: type = default`),
  and set via keyword (or positional) arguments when you instantiate:
  `Trainer(model=my_model, optimizer=O.Adam(lr=1e-3))`.
- **`setup()`** runs once, immediately after fields are assigned. Use
  it for anything that needs to happen before training starts —
  initializing optimizer state, resuming a checkpoint (see §18).
- **`run()`** is called automatically right after `setup()`, but only
  if your subclass actually overrides it. This means instantiating a
  `Trainer` is enough to start training:

  ```python
  trainer = Trainer(model=my_model, optimizer=O.Adam(lr=1e-3))
  # setup() then run() already happened by the time this line finishes
  ```

  A `Struct` that isn't a runnable process (a dataset, a config
  bundle) simply doesn't override `run`, and nothing extra happens.
- **`key=`** is an optional constructor argument. If given, it enables
  `self.rng()` inside `setup()`/`run()` (splits a fresh key each
  call). Calling `self.rng()` without having passed `key=` raises
  `RuntimeError` — there's no silent fallback, since that would hide
  non-determinism.

  ```python
  class Trainer(W.Struct):
      model: L.Dense = None

      def setup(self):
          self.model = L.Dense(3, 3, key=self.rng())

  trainer = Trainer(key=jax.random.PRNGKey(0))
  ```

- **Composition, not inheritance.** A `Struct` holds other `Struct`s
  and `Module`s as fields — a `Trainer` holds a `model`, an
  `optimizer`, maybe a `data` struct — rather than subclassing a
  framework `Trainer` base class. There is no built-in `Train`/
  `Trainer` class to import; you write the `Struct` subclass yourself,
  as shown throughout this guide.

---

## 14. `W.loop` — running the training loop

`xera.weave.loop` is a plain function, not a class. It runs
`jax.lax.scan` or `jax.lax.fori_loop` under the hood and hands back
`(final_carry, outputs)`.

```python
def step(carry, x):
    new_carry = carry + x
    return new_carry, new_carry   # (new_carry, per-step output)

final_carry, outputs = W.loop(step, init_carry=0, steps=5)
```

Signature: `loop(body_fn, init_carry, xs=None, type="scan", steps=1000, stop=None)`

- **`body_fn(carry, x) -> (new_carry, output)`** — called once per
  step. `carry` is whatever state you're threading through (typically
  `(model, opt_state)`, or more if you're also threading a log buffer
  — see §19). `output` is collected into `outputs` across all steps.
- **`xs`** — optional per-step inputs (e.g. a batch index, or actual
  data slices). Defaults to `jnp.arange(steps)`.
- **`type`** — `"scan"` (default) or `"fori_loop"`.
- **`stop`** — optional early-stopping condition, `stop(carry, x) ->
  bool`. See §19 (`W.Callback.early_stopping`, `W.Callback.nan`). Once
  it first fires, every remaining step takes a cheap no-op branch
  instead of running `body_fn` — `steps` itself stays fixed (`jit`
  needs a static trip count), but the remaining compute is skipped.

A minimal but complete training step, threading a model and optimizer
state through `carry`:

```python
def step(carry, i):
    model, opt_state = carry
    def loss_fn(m):
        return jnp.mean((m(x) - y) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model, step=i)
    model = O.apply_updates(model, updates)
    return (model, opt_state), loss

(final_model, final_opt_state), losses = W.loop(
    step, (model, optimizer.init(model)), steps=1000,
)
```

Switching execution strategy is just the `type=` argument — same
`body_fn`, no other changes needed:

```python
final_carry, outputs = W.loop(step, init_carry, steps=1000, type="fori_loop")
```

---

## 15. `W.Loss` — loss functions

`xera.weave.loss.Loss` is a stateless namespace of `@staticmethod`
loss functions — not a class you instantiate, call methods directly
on the class:

```python
loss = W.Loss.L2(pred, target)
loss = W.Loss.CE(logits, labels)
```

Regression: `L1`, `L2`, `RMSE`, `Huber`, `SmoothL1`, `LogCosh`,
`Quantile(pred, target, quantile=0.5)`, `Poisson`, `Gamma`.

Classification: `CE(logits, labels, axis=-1)`, `BCE`,
`BCEWithLogits` (alias of `BCE`), `NLL(log_probs, labels, axis=-1)`,
`KLDiv(log_probs, target_probs, axis=-1)`, `Hinge(pred, target,
margin=1.0)`, `FocalLoss(logits, labels, alpha=0.25, gamma=2.0,
axis=-1)`, `SigmoidFocalCrossEntropy(logits, labels, alpha=0.25,
gamma=2.0)`.

Metric learning / ranking: `CosineEmbedding(pred1, pred2, target,
margin=0.0)`, `MarginRanking(pred1, pred2, target, margin=1.0)`,
`TripletLoss(anchor, positive, negative, margin=1.0)`,
`ContrastiveLoss(pred1, pred2, target, margin=1.0)`.

For every classification loss that takes `labels`, `labels` can be
either integer class indices or already-one-hot vectors — detected by
comparing `labels.ndim` against `logits`/`log_probs.ndim`.

```python
def loss_fn(model, x, y):
    return W.Loss.CE(model(x), y)   # y as class indices or one-hot, either works

loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
```

---

## 16. `O` — optimizers

`xera.weave.optimizer` provides the optimizer implementations and the
composable wrappers around them.

### 4.1 Base optimizers

All follow the same two-method interface — `init(params) -> state` and
`update(grads, state, params=None, step=None) -> (updates, new_state)`:

```python
optimizer = O.Adam(lr=1e-3)
opt_state = optimizer.init(model)
updates, opt_state = optimizer.update(grads, opt_state, model, step=i)
model = O.apply_updates(model, updates)
```

Available: `SGDMomentum`, `Adam`, `AdamW`, `Lion`, `Muon` (and
`MuonCore`), `RMSprop`, `Adagrad`, `Adan`, `Adafactor`, `Shampoo`.

`O.apply_updates(params, updates)` is the plain `params + updates`
pytree add — every optimizer's `update()` returns updates meant to be
applied this way.

### 4.2 Wrappers — composed by calling them on an optimizer

`Clip`, `Schedule`, `Accumulate`, `WeightDecay`, `EMA`, `Freeze`,
`Lookahead`, `Cast` all follow the same pattern: construct the
wrapper, then call it on an inner optimizer to get a wrapped one.

```python
optimizer = O.Clip(threshold=1.0)(O.Adam(lr=1e-3))
optimizer = O.Schedule(fn=my_lr_schedule)(O.Adam(lr=1.0))
```

Wrappers compose by nesting calls:

```python
optimizer = O.Clip(threshold=1.0)(O.Schedule(fn=warmup_cosine)(O.AdamW(lr=1.0)))
```

### 4.3 `O.Partition` — different optimizers for different parameters

`Partition` takes a sequence of `(predicate, optimizer)` rules,
matched against each parameter leaf's path — the first matching rule
wins. Always end with a catch-all (`lambda path, leaf: True`):

```python
optimizer = O.Partition([
    (lambda path, leaf: leaf.ndim == 2, O.Adam(lr=1e-3)),       # matrices
    (lambda path, leaf: True, O.SGDMomentum(lr=1e-2)),          # everything else
])
```

---

## 17. `W.shard` — device sharding

`xera.weave.shard` is a decorator that shards a function's arguments
across multiple devices before calling it, using `jax.device_put` +
`jax.sharding.NamedSharding` under the hood. It exists so a function
can declare its sharding intent inline, without hand-building a
`Mesh` elsewhere in the script — everything else in `weave` (the loop,
the optimizers, the callbacks) assumes a single logical device;
`shard` is the one piece concerned with *where* an array actually
lives.

### Where it goes: shard the batch, at the top of the step

The most common placement is data-parallel training: shard each
step's batch across a `'data'` axis right where the batch is
produced/consumed, inside the `body_fn` you pass to `W.loop`. This
works whether or not that `body_fn` ends up under an outer
`jax.jit` — `shard` calls `jax.device_put` with a `Sharding` object,
which JAX honors as a sharding constraint even from inside a traced
`jax.lax.scan`/`fori_loop` step, no separate `jax.jit` required:

```python
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from xera.weave import shard

@shard(P('data', None), P('data', None))
def shard_batch(x, y):
    return x, y

class Trainer(W.Struct):
    model: L.Dense = None
    optimizer: O.Adam = None
    steps: int = 1000

    def setup(self):
        self.opt_state = self.optimizer.init(self.model)

    def step(self, carry, i):
        model, opt_state = carry
        x, y = get_batch(i)          # your data-loading, shape (batch, features)
        x, y = shard_batch(x, y)     # <-- sharded here, before it's used below

        def loss_fn(m):
            return W.Loss.L2(m(x), y)

        loss, grads = jax.value_and_grad(loss_fn)(model)
        updates, opt_state = self.optimizer.update(grads, opt_state, model, step=i)
        model = O.apply_updates(model, updates)
        return (model, opt_state), loss

    def run(self):
        (self.model, self.opt_state), losses = W.loop(
            self.step, (self.model, self.opt_state), steps=self.steps,
        )
```

`model`/`opt_state` are left un-sharded here (each device gets its own
full copy) — only the batch (`x`, `y`) is split across devices along
its leading (batch) dimension, each device computing gradients on its
own shard. On a single device this exact code still runs unmodified
(sharding is skipped, see below); scaling to more devices needs no
code change, just more devices visible to `jax.devices()`.

The same decorator also composes with an explicitly `@jax.jit`'d
function, if you're jitting a step/forward function yourself outside
`W.loop` (e.g. calling it directly in a plain Python loop rather than
via `jax.lax.scan`):

```python
import jax
from jax.sharding import PartitionSpec as P
from xera.weave import shard

@jax.jit
@shard(P('data', None), P(None, 'model'))
def forward(x, w):
    return x @ w
```

- Pass a `PartitionSpec` per **positional** argument (`*specs`, same
  order as the wrapped function's parameters) and/or per **keyword**
  argument (`**kwspecs`, keyed by name). `None` (or simply omitting a
  trailing argument) leaves that argument un-sharded.
- The device mesh is built automatically from `jax.devices()`, with
  axis names inferred straight from the specs you pass in — no
  `Mesh(...)` call needed anywhere in user code. Each argument is
  sharded right before the wrapped function runs.
- **Single device visible:** sharding is skipped and the function runs
  unmodified — a `UserWarning` fires once per process (not once per
  call), since scripts should still run as-is on a laptop.
- **Multiple devices visible but a shape doesn't evenly divide** the
  way its spec asks for: raises a clear `ValueError` naming the
  function, the offending argument, its shape, the spec, and the mesh
  — not a raw internal `jax` traceback.

---

## 18. `S` — checkpointing

`xera.serialize` handles saving/loading either just a model, or a
model + optimizer state + arbitrary metadata together as one `.sxera`
file. The typical training pattern needs only two calls: one in
`setup()`, one in your step function.

### 6.1 `S.save_model` / `S.load_model` — model only

For when you only need the model's parameters, not a full training
checkpoint — e.g. exporting a trained model for inference.

```python
S.save_model(model, "model.safetensors")
model = S.load_model(template, "model.safetensors")
```

- `save_model(module, path)` flattens `module`'s pytree and writes
  every leaf to an ordinary `.safetensors` file, keyed by attribute
  path (e.g. `"weight"`, `"blocks.0.dense.weight"`).
- `load_model(template, path)` reads that file back and reshapes/
  casts each tensor to match `template`'s corresponding leaf — so
  `template` must have the same architecture (shapes and dtypes) as
  the model that was saved, typically a freshly constructed instance:
  `S.load_model(L.Dense(4, 8), "model.safetensors")`.
- The file is a plain safetensors file — openable with any standard
  safetensors reader, not just `xera`.

### 6.2 `S.save_struct` — model + optimizer + metadata, one-shot save

`save_struct(model, optimizer, metadata, path)` writes a model, an
optimizer's state, and arbitrary metadata (step counters, RNG keys,
config, ...) together into one `.sxera` file — itself an ordinary
safetensors file, just with three key prefixes (`model.`,
`optimizer.`, `meta.`) to keep the three components apart, plus a
JSON snapshot of `metadata`'s non-array values and a `repr()` of each
part's tree structure (used for the drift-detection in `load_struct`
below).

```python
S.save_struct(
    model, optimizer,
    metadata={"step": 1000, "key": rng_key},
    path="ckpt.sxera",
)
```

This is a one-shot, untraced save — call it directly from plain
Python. For saving *inside* a traced training step every N steps, use
`S.checkpointer` (§18.4) instead, which wraps this same format.

`Struct` also exposes this as a convenience instance method,
`self.save_struct(model, optimizer, metadata, path)` — identical
behavior, just callable on any `Struct` (e.g. your `Trainer`) without
a separate `S.` import:

```python
class Trainer(W.Struct):
    def checkpoint_now(self):
        self.save_struct(self.model, self.opt_state, {"step": self.step}, "ckpt.sxera")
```

### 6.3 `S.load_struct` — auto-resume, called once in `setup()`

`load_struct(model_template, optimizer_template, metadata_template, path, release=False)`

`path` can be either an exact `.sxera` file (strict — raises if
missing), or a **directory** (auto-discovery mode — this is what you
want for a `Trainer`'s `path` field):

1. A `.sxera` file inside it → full load (model + optimizer +
   metadata).
2. No `.sxera` but a `.safetensors` file → model loaded from it,
   optimizer/metadata returned as given (a plain `.safetensors` file
   has no state for them).
3. Neither → all three templates returned unchanged (fresh start).

This means the same call trains from scratch on a first run and
resumes automatically on every run after, given nothing but a
directory:

```python
def setup(self):
    self.model, self.opt_state, self.meta = S.load_struct(
        self.model, self.optimizer.init(self.model),
        {"step": jnp.asarray(0)}, self.path,
    )
```

> **Metadata dtype note:** if a metadata field will also flow through
> `checkpointer` (§18.4) from inside a traced step, it round-trips as a
> JAX array (not a plain Python value), because `io_callback` coerces
> every argument leaf into an array. So the template you pass to
> `load_struct` needs matching array-shaped leaves too —
> `{"step": jnp.asarray(0)}`, not `{"step": 0}` — or the checkpointed
> value won't overwrite the template's on load.

`release=True` treats a structural/config mismatch (e.g. you changed
an optimizer's hyperparameters, or added a model layer) as intentional
instead of raising `ValueError`.

### 6.4 `S.checkpointer` — auto-save, called every step

`checkpointer(path, name="ckpt", every=1, override=True)` returns a
`save(model, optimizer, metadata, step)` function, meant to be called
directly from inside a traced step:

```python
def setup(self):
    ...
    self.save = S.checkpointer(self.path, every=100)

def step(self, carry, i):
    model, opt_state = carry
    ...
    self.save(model, opt_state, {"step": i}, i)
    return (model, opt_state), loss
```

- **`every`** throttles how often it actually writes to disk
  (`step % every == 0`), implemented with `jax.lax.cond` — a skipped
  step costs only a scalar comparison, nothing ever reaches disk.
- **`override`** controls retention: `True` (default) keeps only the
  single latest checkpoint file in `path` (deletes the rest after each
  write); `False` keeps every checkpoint ever written (full history).
- `save` wraps its own `jax.experimental.io_callback` internally — you
  never need to reach for `Callback.io` yourself just to checkpoint.

### 6.5 `S.extract_model` — pull just the model out of a `.sxera`

`extract_model(sxera_path, out_path)` re-keys the `model.`-prefixed
tensors of a `.sxera` checkpoint into a plain `model.safetensors`
file — no template required, since `.sxera` keys are already
fully-qualified paths. Useful once training's done and you only want
to ship/deploy the model, not the optimizer state or metadata that
came along for resuming.

```python
S.extract_model("ckpt.sxera", "model.safetensors")
model = S.load_model(L.Dense(4, 8), "model.safetensors")  # ordinary load, §18.1
```

### 6.6 Putting it together

```python
class Trainer(W.Struct):
    model: L.Dense = None
    optimizer: O.Adam = None
    path: str = "runs/my_model"
    steps: int = 1000

    def setup(self):
        self.model, self.opt_state, self.meta = S.load_struct(
            self.model, self.optimizer.init(self.model),
            {"step": jnp.asarray(0)}, self.path,
        )
        self.save = S.checkpointer(self.path, every=100)

    def step(self, carry, i):
        model, opt_state = carry
        loss, grads = jax.value_and_grad(self.loss_fn)(model)
        updates, opt_state = self.optimizer.update(grads, opt_state, model, step=i)
        model = O.apply_updates(model, updates)
        self.save(model, opt_state, {"step": i}, i)
        return (model, opt_state), loss

    def run(self):
        (self.model, self.opt_state), losses = W.loop(
            self.step, (self.model, self.opt_state), steps=self.steps,
        )

# First run: trains from scratch, writes checkpoints to runs/my_model/.
# Every run after: automatically resumes from the latest checkpoint there.
trainer = Trainer(model=L.Dense(3, 3, key=jax.random.PRNGKey(0)), optimizer=O.Adam(lr=1e-3))
```

---

## 19. `W.Callback` — side-effects and stop conditions

`Callback` is a stateless namespace (all `@staticmethod`), not a class
you subclass. Everything here is called from inside a traced `step`
function, unless noted otherwise.

### 7.1 `Callback.print` — debug printing with throttling

```python
Callback.print(i, "loss={loss} lr={lr}", every=10, loss=loss, lr=lr)
```

- The format string uses `str.format`-style placeholders, **not an
  f-string** — `loss`/`lr` are traced values, so they can't be
  rendered into a string before `Callback.print` runs; `jax.debug.print`
  needs to receive them separately to defer rendering to runtime.
- `fmt` is optional — omit it to print every `**values` as `name=value`.
- `every` throttles printing via `jax.lax.cond`, same mechanism as
  `checkpointer`'s `every`.

### 7.2 `Callback.io` — arbitrary Python side-effects

For anything not covered by `print`/`checkpointer`/`log` — a plain
`jax.experimental.io_callback` wrapper:

```python
Callback.io(i, my_python_fn, arg1, arg2, kwarg=value)
```

### 7.3 `Callback.log` — buffered metric logging

Per-step file writes are too slow to do every step, so `Callback.log`
batches values in memory and flushes to a `.jsonl` file only once
every `every` steps. Because `jax.lax.scan`/`fori_loop` are purely
functional, that buffer can't be hidden state — it has to be threaded
through your own `carry` explicitly, like any other value:

```python
def setup(self):
    self.log_fn, self.log_buffer0 = W.Callback.log(
        self.path, every=50, loss=jnp.float32, lr=jnp.float32,
    )

def step(self, carry, i):
    model, opt_state, log_buffer = carry
    ...
    log_buffer = self.log_fn(log_buffer, i, loss=loss, lr=lr)
    return (model, opt_state, log_buffer), loss

def run(self):
    init = (self.model, self.opt_state, self.log_buffer0)
    (final_model, final_opt_state, _), losses = W.loop(self.step, init, steps=1000)
```

- Writes to `{path}/{name}.jsonl` (default `name="log"`) — same
  directory you pass to `checkpointer`/`load_struct`, so a run's
  checkpoint and its logs live side by side.
- **Always append-only.** Every flush appends new lines to the same
  file; it's never truncated or overwritten automatically — including
  across a fresh restart from step 0 with the same `path`. If you want
  a clean log for a new run, clear the directory yourself first.
- Every call must log exactly the fields declared up front (`loss`,
  `lr`, ...) — a missing or extra field raises `ValueError`.

### 7.4 Stop conditions — for `W.loop(..., stop=...)`

`Callback.early_stopping(patience, extract)` and `Callback.nan()` are
factories: calling them returns a `stop_fn(carry, x) -> bool`, meant
to be passed as `loop`'s `stop=` argument (§14), not called directly
inside your step.

```python
stop = W.Callback.early_stopping(patience=10, extract=lambda carry: carry.since_improved)
final, outputs = W.loop(step, init_carry, steps=1000, stop=stop)

# or: stop once any float leaf in carry goes NaN/Inf
final, outputs = W.loop(step, init_carry, steps=1000, stop=W.Callback.nan())
```

---

## 20. Full example

```python
import jax
import jax.numpy as jnp
import xera.loom as L
import xera.weave as W
import xera.weave.optimizer as O
import xera.serialize as S


class Trainer(W.Struct):
    model: L.Dense = None
    optimizer: O.Optimizer = None
    path: str = "runs/demo"
    steps: int = 500

    def setup(self):
        self.model, self.opt_state, self.meta = S.load_struct(
            self.model, self.optimizer.init(self.model),
            {"step": jnp.asarray(0)}, self.path,
        )
        self.save = S.checkpointer(self.path, every=100)
        self.log_fn, self.log_buffer0 = W.Callback.log(
            self.path, every=50, loss=jnp.float32,
        )

    def loss_fn(self, model, x, y):
        return W.Loss.L2(model(x), y)

    def step(self, carry, i):
        model, opt_state, log_buffer = carry
        x = jnp.ones((8, 4))
        y = jnp.zeros((8, 4))

        loss, grads = jax.value_and_grad(self.loss_fn)(model, x, y)
        updates, opt_state = self.optimizer.update(grads, opt_state, model, step=i)
        model = O.apply_updates(model, updates)

        W.Callback.print(i, "loss={loss}", every=10, loss=loss)
        self.save(model, opt_state, {"step": i}, i)
        log_buffer = self.log_fn(log_buffer, i, loss=loss)

        return (model, opt_state, log_buffer), loss

    def run(self):
        init = (self.model, self.opt_state, self.log_buffer0)
        (self.model, self.opt_state, _), losses = W.loop(
            self.step, init, steps=self.steps,
        )


key = jax.random.PRNGKey(0)
trainer = Trainer(model=L.Dense(4, 4, key=key), optimizer=O.Adam(lr=1e-3))
# runs/demo/ now has: the latest ckpt_*.sxera and log.jsonl.
# Re-running the same script picks up training from there automatically.
```
