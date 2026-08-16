

from __future__ import annotations
import jax


def _default_emit(name, step, value):
    if step is None:
        print(f"{name} = {float(value):.6f}")
    else:
        print(f"[step {int(step)}] {name} = {float(value):.6f}")


class Metrics:
    """Registry of side-effect functions runnable from inside `jax.jit`/
    `lax.scan` training steps. Register once, call `Metrics.log(...)`
    from the step function -- it's traced normally, the registered
    functions only run host-side via `jax.debug.callback`.

        @Metrics.register("loss")
        def _(step, value):
            print(f"step {step}: loss={value:.4f}")

        Metrics.log(step, loss=loss, acc=acc)   # inside a scan/jit step
    """

    _registry = {}

    @classmethod
    def register(cls, name, fn=None):
        def deco(f):
            cls._registry[name] = f
            return f
        if fn is not None:
            cls._registry[name] = fn
            return fn
        return deco

    @classmethod
    def unregister(cls, name):
        cls._registry.pop(name, None)

    @classmethod
    def log(cls, step=None, **values):
        def _emit(step, values):
            for name, value in values.items():
                fn = cls._registry.get(name, None)
                if fn is not None:
                    fn(step, value)
                else:
                    _default_emit(name, step, value)
        jax.debug.callback(_emit, step, values)


__all__ = ["Metrics"]
