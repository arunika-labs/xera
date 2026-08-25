# xera.weave — API Guide

This guide covers the training side of xera: `xera.weave` (loop,
callbacks, optimizers) and `xera.serialize` (checkpointing), and the
`Struct` base class that ties them together into a `Trainer`.

Conventional aliases used throughout this guide and recommended for
your own code:

```python
import xera.weave as W
import xera.weave.optimizer as O
import xera.serialize as S
```

`xera.loom` (`L`, layers/modules) and `xera.loom.functional` (`F`,
functional ops) come up occasionally below wherever a model is needed,
but this guide is about `weave`, not about building models.

```python
import xera.loom as L            # occasional, for model definitions
import xera.loom.functional as F # occasional, for functional ops
```

---

## 1. `Struct` — the base of everything on the training side

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
  initializing optimizer state, resuming a checkpoint (see §4).
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

## 2. `W.loop` — running the training loop

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
  — see §6). `output` is collected into `outputs` across all steps.
- **`xs`** — optional per-step inputs (e.g. a batch index, or actual
  data slices). Defaults to `jnp.arange(steps)`.
- **`type`** — `"scan"` (default) or `"fori_loop"`.
- **`stop`** — optional early-stopping condition, `stop(carry, x) ->
  bool`. See §5 (`W.Callback.early_stopping`, `W.Callback.nan`). Once
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

## 3. `O` — optimizers

`xera.weave.optimizer` provides the optimizer implementations and the
composable wrappers around them.

### 3.1 Base optimizers

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

### 3.2 Wrappers — composed by calling them on an optimizer

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

### 3.3 `O.Partition` — different optimizers for different parameters

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

## 4. `S` — checkpointing

`xera.serialize` handles saving/loading a model, optimizer state, and
arbitrary metadata together, as `.sxera` files. The typical pattern
needs only two calls: one in `setup()`, one in your step function.

### 4.1 `S.load_struct` — auto-resume, called once in `setup()`

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
> `checkpointer` (§4.2) from inside a traced step, it round-trips as a
> JAX array (not a plain Python value), because `io_callback` coerces
> every argument leaf into an array. So the template you pass to
> `load_struct` needs matching array-shaped leaves too —
> `{"step": jnp.asarray(0)}`, not `{"step": 0}` — or the checkpointed
> value won't overwrite the template's on load.

`release=True` treats a structural/config mismatch (e.g. you changed
an optimizer's hyperparameters, or added a model layer) as intentional
instead of raising `ValueError`.

### 4.2 `S.checkpointer` — auto-save, called every step

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

### 4.3 Putting it together

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

## 5. `W.Callback` — side-effects and stop conditions

`Callback` is a stateless namespace (all `@staticmethod`), not a class
you subclass. Everything here is called from inside a traced `step`
function, unless noted otherwise.

### 5.1 `Callback.print` — debug printing with throttling

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

### 5.2 `Callback.io` — arbitrary Python side-effects

For anything not covered by `print`/`checkpointer`/`log` — a plain
`jax.experimental.io_callback` wrapper:

```python
Callback.io(i, my_python_fn, arg1, arg2, kwarg=value)
```

### 5.3 `Callback.log` — buffered metric logging

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

### 5.4 Stop conditions — for `W.loop(..., stop=...)`

`Callback.early_stopping(patience, extract)` and `Callback.nan()` are
factories: calling them returns a `stop_fn(carry, x) -> bool`, meant
to be passed as `loop`'s `stop=` argument (§2), not called directly
inside your step.

```python
stop = W.Callback.early_stopping(patience=10, extract=lambda carry: carry.since_improved)
final, outputs = W.loop(step, init_carry, steps=1000, stop=stop)

# or: stop once any float leaf in carry goes NaN/Inf
final, outputs = W.loop(step, init_carry, steps=1000, stop=W.Callback.nan())
```

---

## 6. Full example

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
        return jnp.mean((model(x) - y) ** 2)

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
