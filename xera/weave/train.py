

"""
Training module for JAX-based neural network training.

This module provides a high-level training loop abstraction that handles
optimization, loss computation, and metric logging in a JAX-compatible way.
"""

from __future__ import annotations
import jax
from .state import State
from .loop import Loop
from .optimizer.base import apply_updates
from .metrics import Metrics


class Train(State):
    """
    A high-level training loop for JAX-based neural network training.
    
    This class provides a flexible training framework that handles:
    - Optimizer initialization and parameter updates
    - Loss computation and gradient calculation
    - Optional metric logging during training
    - Support for both scan and fori_loop implementations
    
    Subclasses must implement the loss_fn and get_batch methods to define
    the specific training logic.
    
    Attributes:
        optimizer: An instance of xera.weave.Optimizer for parameter updates.
        steps: The number of training steps to perform.
        loop_type: The type of loop implementation ("scan" or "fori_loop").
        log_every: Logging frequency. 0 disables logging, N logs every N steps.
    
    Example:
        >>> class MyTrainer(Train):
        ...     def loss_fn(self, pred, target):
        ...         return jnp.mean((pred - target) ** 2)
        ...     
        ...     def get_batch(self, i):
        ...         return x_data[i], y_data[i]
        ...
        >>> trainer = MyTrainer(optimizer=Adam(lr=0.001), steps=1000)
        >>> trained_model = trainer(model)
    """

    optimizer: "Optimizer" = None
    steps: int = 100
    loop_type: str = "scan"
    log_every: int = 0   # 0 = no metric logging

    def setup(self):
        """
        Validate configuration and initialize the training loop.
        
        Raises:
            AssertionError: If optimizer is None or loop_type is invalid.
        """
        assert self.optimizer is not None, "Train requires an `optimizer=` parameter (an instance of xera.weave.Optimizer)."
        assert self.loop_type in ("scan", "fori_loop"), f"Train only supports loop_type='scan' or 'fori_loop', got: {self.loop_type}"
        self.loop = Loop(type=self.loop_type, steps=self.steps)

    def loss_fn(self, pred, target):
        """
        Compute the loss between predictions and targets.
        
        This method must be implemented by subclasses to define the
        specific loss function for the training task.
        
        Args:
            pred: Model predictions.
            target: Ground truth targets.
        
        Returns:
            A scalar loss value.
        
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Train subclasses must override loss_fn(pred, target).")

    def get_batch(self, i):
        """
        Get a batch of data for the given training step.
        
        This method must be implemented by subclasses to define how
        training data is retrieved for each step.
        
        Args:
            i: The current training step index.
        
        Returns:
            A tuple (x, y) containing input data and targets.
        
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Train subclasses must override get_batch(i).")

    def step(self, carry, i):
        """
        Perform a single training step.
        
        This method computes gradients, applies optimizer updates, and
        optionally logs metrics. It's designed to be used with JAX's
        scan or fori_loop operations.
        
        Args:
            carry: A tuple (model, opt_state) containing the current model
                and optimizer state.
            i: The current training step index.
        
        Returns:
            A tuple ((model, opt_state), loss) containing the updated
            model and optimizer state, plus the loss value.
        """
        model, opt_state = carry
        x, y = self.get_batch(i)

        def loss_only(m):
            """Helper function to compute loss without gradients."""
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
        """
        Train the model and return the final trained model.
        
        This is a convenience method that initializes the optimizer and
        runs the training loop, returning only the final model.
        
        Args:
            model: The initial model to train.
        
        Returns:
            The trained model after all training steps.
        """
        opt_state = self.optimizer.init(model)
        (final_model, _final_opt_state), _losses = self.loop.run(
            self.step, (model, opt_state)
        )
        return final_model

    def run(self, model):
        """
        Train the model and return the final model, optimizer state, and losses.
        
        This method provides more detailed output than __call__, including
        the final optimizer state and loss values from each step.
        
        Args:
            model: The initial model to train.
        
        Returns:
            A tuple (final_model, final_opt_state, losses) containing:
                - final_model: The trained model
                - final_opt_state: The final optimizer state
                - losses: Loss values from each training step
        """
        opt_state = self.optimizer.init(model)
        (final_model, final_opt_state), losses = self.loop.run(
            self.step, (model, opt_state)
        )
        return final_model, final_opt_state, losses


__all__ = ["Train"]
