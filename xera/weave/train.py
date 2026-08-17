

from __future__ import annotations
import jax
from .state import State
from .loop import Loop
from .optimizer.base import apply_updates
from .metrics import Metrics


class Train(State):

    optimizer: "Optimizer" = None
    steps: int = 100
    loop_type: str = "scan"
    log_every: int = 0   # 0 = no metric logging

    def setup(self):
        assert self.optimizer is not None, "Train wajib diberi `optimizer=` (instance xera.weave.Optimizer)."
        assert self.loop_type == "scan", (
            "Train hanya mendukung loop_type='scan'. Train.run() mengumpulkan "
            "`losses` per step lewat output stack dari lax.scan; lax.fori_loop "
            "tidak mengumpulkan output per-step sama sekali (cuma carry akhir), "
            "jadi tidak kompatibel dengan kontrak Train.run(). Kalau butuh "
            "fori_loop tanpa histori losses, pakai xera.weave.Loop langsung, "
            "bukan lewat Train."
        )
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
        updates, opt_state = self.optimizer.update(grads, opt_state, model, step=i)
        model = apply_updates(model, updates)

        if self.log_every:
            jax.lax.cond(
                i % self.log_every == 0,
                lambda: Metrics.log(i, loss=loss),
                lambda: None,
            )

        return (model, opt_state), loss

    def __call__(self, model):
        opt_state = self.optimizer.init(model)
        (final_model, _final_opt_state), _losses = self.loop.run(
            self.step, (model, opt_state)
        )
        return final_model

    def run(self, model):
        opt_state = self.optimizer.init(model)
        (final_model, final_opt_state), losses = self.loop.run(
            self.step, (model, opt_state)
        )
        return final_model, final_opt_state, losses


__all__ = ["Train"]
