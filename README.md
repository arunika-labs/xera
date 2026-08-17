# xera

A neural network library in JAX, designed for functional programming needs and ease of use with a modern API.

## Install

```bash
pip install git+https://github.com/arunika-labs/xera.git
```

## Quickstart

```python
import jax
import xera.loom as L
import jax.numpy as jnp

model = L.Dense(4, 8, key=jax.random.PRNGKey(0))
x = jnp.ones((2, 4))
y = model(x)
```

## API

Everything lives under four short aliases off the `xera` top level:

```python
import xera

L = xera.loom       # layers / modules
W = xera.weave       # training loop, loss, optimizers, metrics, callbacks
O = xera.weave.optimizer  # also reachable as xera.O
S = xera.serialize    # save / load (safetensors)
```

### `L` — layers

```python
import xera.loom as L

model = L.Dense(4, 8, key=jax.random.PRNGKey(0))
y = model(x)
```

### `W` — training

```python
import xera.weave as W

class Trainer(W.Train):
    def loss_fn(self, pred, target):
        return W.Loss.L2(pred, target)

    def get_batch(self, i):
        return x_data[i], y_data[i]

trainer = Trainer(optimizer=W.Adam(lr=1e-3), steps=1000)
trained_model = trainer(model)
```

### `O` — optimizers

```python
import xera.weave.optimizer as O

opt = O.Adam(lr=1e-3)
opt_state = opt.init(model)
updates, opt_state = opt.update(grads, opt_state, model)
```

### `S` — serialize

```python
import xera.serialize as S

S.save_model(model, "model.safetensors")
model = S.load_model(template, "model.safetensors")
```

## License

Apache-2.0
