from jax import numpy as jnp
import jax

class AttentionTrackerBase:
    """
    Base class for tracking attention weights in a neural network.
    """

    @staticmethod
    def init_params(*args):
        """
        Initialize the AttentionTrackerBase.
        """
        # define static parameters
        static_params = {
            "lr": None,
            "decay": None,
        }
        raise NotImplementedError

    @staticmethod
    def step(*args):
        """
        Steps attention weights
        
        Args:
            weights: attention weights to be updated.
            hyperparams: dictionary of hyperparameters for the update step.

        Returns:
            new attention weights.
        """
        raise NotImplementedError

    @staticmethod
    def initialise_weights(shape):
        """
        Initialise attention weights.

        Args:
            N: number of neurons.
            out_dim: output dimension.

        Returns:
            Initialised attention weights.
        """
        return jnp.zeros(shape)
    
# class RBAttention(AttentionTrackerBase):
#     """
#     Class for tracking attention weights using the Recurrent Backpropagation Attention method.
#     """

#     @staticmethod
#     def init_params(N, out_dim, eta_lr=1e-2, gamma_decay=0.99):
#         """
#         Initialize the RBAttention tracker.
#         """
#         static_params = {
#             "eta_lr": eta_lr,
#             "gamma_decay": gamma_decay,
#         }
#         trainable_params = {
#             "alpha": AttentionTrackerBase.initialise_weights(N, out_dim),
#         }
#         return static_params, trainable_params

#     @staticmethod
#     def step(weights, residuals, hyperparams):
#         """
#         Take step based on RBA method

#         attn_new = decay * attn_old + lr * (|residual_i|/max_i(|residual_i|))
#         """
#         attention_old = weights
#         r_abs     = jnp.abs(residuals)
#         r_max     = jnp.max(r_abs)
#         attention_new = hyperparams["gamma_decay"]*attention_old + hyperparams["eta_lr"]*(r_abs/(r_max+1e-12))
#         return attention_new

class RBAttention(AttentionTrackerBase):
    """
    Residual-Based Attention tracker.
    αᵢ ← γ αᵢ + η · |rᵢ| / max_j |rⱼ|
    """

    @staticmethod
    def initialise_weights(shape, init_val=1.0, dtype=jnp.float32):
        """Return an array filled with `init_val`."""
        return jnp.full(shape, init_val, dtype=dtype)
    
    @staticmethod
    def init_params(shape, eta_lr=1e-2, gamma_decay=0.99, init_val=1.0):
        static_params = dict(
            eta_lr=eta_lr, 
            gamma_decay=gamma_decay,
            clip_min=0.0, 
            clip_max=1.0
            )
        trainable = dict(
            alpha=RBAttention.initialise_weights(shape, init_val)
            )
        return static_params, trainable

    @staticmethod
    @jax.jit
    def step(alpha_old, residuals, hyperparams):
        """
        α_new = γ α_old + η (|r| / max|r|); clipped to [clip_min, clip_max]
        Args
        ----
        alpha_old : (...,)          current attention weights
        residuals : same shape      current residuals
        hyper     : dict            needs keys 'eta_lr', 'gamma_decay',
                                    'clip_min', 'clip_max'
        """
        r_abs   = jnp.abs(residuals)
        r_max   = jnp.maximum(jnp.max(r_abs), 1e-12)
        alpha_new = (hyperparams["gamma_decay"] * alpha_old + hyperparams["eta_lr"] * (r_abs / r_max))
        return jnp.clip(alpha_new, hyperparams["clip_min"], hyperparams["clip_max"])
