# xera

A neural network library in JAX, designed for functional programming needs and ease of use with a modern API.

## Install

pypi
```bash
pip install xera
```
github
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

Everything lives under a few short aliases off the `xera` top level:

```python
import xera.loom as L               # layers / modules
import xera.loom.functional as F    # activations + functional ops
import xera.weave as W              # training loop, loss, callbacks, sharding
import xera.weave.optimizer as O    # optimizers
import xera.serialize as S          # save / load (safetensors)
```

The full API reference — every layer, every training/optimizer/loss
utility, sharding, checkpointing, and a complete end-to-end training
example — lives in **[`docs/API.md`](docs/API.md)**. Start there for
anything beyond this quickstart.

## License

Apache-2.0
