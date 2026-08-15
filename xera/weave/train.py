

from __future__ import annotations
import jax
import optax
from .state import State
from .loop import Loop


class Train(State):
    

    lr: float = 1e-3
    steps: int = 100
    loop_type: str = "scan"   

    def setup(self):
        
        
        
        object.__setattr__(self, "_optax", optax.adam(self.lr))
        self.loop = Loop(type=self.loop_type, steps=self.steps)

    
    def loss_fn(self, pred, target):
        raise NotImplementedError("Subclass Train wajib override loss_fn(pred, target).")

    def get_batch(self, i):
        
        raise NotImplementedError("Subclass Train wajib override get_batch(i).")

    
    def step(self, carry, i):
        
        model, opt_state = carry
        x, y = self.get_batch(i)

        def loss_only(m):
            pred = m(x)  
            return self.loss_fn(pred, y)

        loss, grads = jax.value_and_grad(loss_only)(model)
        updates, opt_state = self._optax.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)
        return (model, opt_state), loss

    def __call__(self, model):
        
        opt_state = self._optax.init(model)
        (final_model, _final_opt_state), _losses = self.loop.run(
            self.step, (model, opt_state)
        )
        return final_model

    def run(self, model):
        
        opt_state = self._optax.init(model)
        (final_model, _final_opt_state), losses = self.loop.run(
            self.step, (model, opt_state)
        )
        return final_model, losses


__all__ = ["Train"]