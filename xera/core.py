

from __future__ import annotations
import dataclasses
import jax
import jax.numpy as jnp


class RNGPool:
    
    __slots__ = ("_key",)

    def __init__(self, key):
        self._key = key

    def next(self):
        self._key, sub = jax.random.split(self._key)
        return sub

    def split(self, n):
        self._key, *subs = jax.random.split(self._key, n + 1)
        return subs


class State:
    
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"State({self.value!r})"


jax.tree_util.register_pytree_node(
    State,
    lambda s: ((s.value,), None),
    lambda aux, children: State(children[0]),
)


class Module:
    

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls = dataclasses.dataclass(cls, eq=False, repr=False, init=False)
        jax.tree_util.register_pytree_node(
            cls,
            cls._tree_flatten,
            cls._tree_unflatten,
        )

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args, key=None, **kwargs):
        field_names = [f.name for f in dataclasses.fields(self)]
        positional = dict(zip(field_names, args))
        for name, val in {**positional, **kwargs}.items():
            object.__setattr__(self, name, val)

        
        
        if key is not None:
            object.__setattr__(self, "_rng_pool", RNGPool(key))
        self.setup()
        if key is not None:
            object.__delattr__(self, "_rng_pool")  

    def setup(self):
        
        pass

    def rng(self, n=None):
        
        pool = getattr(self, "_rng_pool", None)
        if pool is None:
            raise RuntimeError(
                "self.rng() dipanggil tapi Module ini dibuat tanpa `key=`."
            )
        return pool.split(n) if n is not None else pool.next()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    
    def params_dict(self):
        
        out = {}
        for name, val in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(val, Module):
                out[name] = val.params_dict()
            elif isinstance(val, jnp.ndarray):
                out[name] = val
            elif isinstance(val, State):
                continue  
            elif isinstance(val, (list, tuple)) and val and isinstance(val[0], Module):
                out[name] = [v.params_dict() for v in val]
        return out

    def state_dict(self):
        
        out = {}
        for name, val in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(val, Module):
                sub = val.state_dict()
                if sub:
                    out[name] = sub
            elif isinstance(val, State):
                out[name] = val.value
        return out

    
    def _tree_flatten(self):
        dynamic_names, dynamic_vals = [], []
        static_names, static_vals = [], []
        for name, val in self.__dict__.items():
            if isinstance(val, (jnp.ndarray, Module, State)) or val is None:
                dynamic_names.append(name)
                dynamic_vals.append(val)
            elif isinstance(val, list) and val and all(isinstance(v, Module) for v in val):
                dynamic_names.append(name)
                dynamic_vals.append(val)
            else:
                static_names.append(name)
                static_vals.append(val)
        aux_data = (tuple(dynamic_names), tuple(static_names), tuple(static_vals))
        return tuple(dynamic_vals), aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        dynamic_names, static_names, static_vals = aux_data
        obj = object.__new__(cls)
        for name, val in zip(dynamic_names, children):
            object.__setattr__(obj, name, val)
        for name, val in zip(static_names, static_vals):
            object.__setattr__(obj, name, val)
        return obj

    def __repr__(self):
        field_names = [f.name for f in dataclasses.fields(self)]
        parts = ", ".join(f"{n}={getattr(self, n)!r}" for n in field_names)
        return f"{type(self).__name__}({parts})"


def param(key, init_fn, shape, dtype=jnp.float32):
    
    return init_fn(key, shape, dtype)