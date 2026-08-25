# xera.loom — API Guide

This guide covers the model-building side of xera: `xera.loom` (layers)
and `xera.loom.functional` (activations + attention), and the `Module`
base class every layer is built on. For the training side (`Struct`,
`W.loop`, optimizers, checkpointing, callbacks), see `docs/API.md`.

Conventional aliases used throughout this guide and recommended for
your own code:

```python
import xera.loom as L
import xera.loom.functional as F
```

Everything in this guide is a `Module`: a dataclass-like, JAX-pytree
layer with parameters, composed by nesting (a `TransformerBlock` holds
a `MultiHeadAttention` and an `MLP`, an `MLP` holds two `Dense`, and so
on). Nothing here needs `jax.jit`/`vmap`/`grad` applied manually inside
this guide's examples — layers are just called like functions
(`layer(x)`) and compose with JAX transforms the ordinary way from the
outside.

---

## 1. `Module` — the base every layer is built on

`xera.core.Module` (re-exported as `xera.loom.Module`, alongside
`Buffer`, `RNGPool`, and `param`) is what `Dense`, `Conv`,
`MultiHeadAttention`, and every other layer in this guide subclasses.
You'll want it directly as soon as you write a custom layer.

```python
from xera.loom import Module, param
import xera.initializers as init

class MyLinear(Module):
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
  (`docs/API.md` §6.3) detect an accidentally-changed architecture.

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

`docs/API.md` §5 covers `xera.weave.shard` for **data-parallel**
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
skips sharding entirely, per `docs/API.md` §5); on multiple devices,
combine this with §5's data-parallel batch example if you want both a
sharded batch *and* sharded model weights in the same training step.
