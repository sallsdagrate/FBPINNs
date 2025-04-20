from jax import numpy as jnp

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
    def initialise_weights(N, out_dim):
        """
        Initialise attention weights.

        Args:
            N: number of neurons.
            out_dim: output dimension.

        Returns:
            Initialised attention weights.
        """
        return jnp.zeros((N, out_dim))
    
class RBAttention(AttentionTrackerBase):
    """
    Class for tracking attention weights using the Recurrent Backpropagation Attention method.
    """

    @staticmethod
    def init_params(N, out_dim, eta_lr=1e-2, gamma_decay=0.99):
        """
        Initialize the RBAttention tracker.
        """
        static_params = {
            "eta_lr": eta_lr,
            "gamma_decay": gamma_decay,
        }
        trainable_params = {
            "alpha": AttentionTrackerBase.initialise_weights(N, out_dim),
        }
        return static_params, trainable_params

    @staticmethod
    def step(weights, residuals, hyperparams):
        """
        Take step based on RBA method

        attn_new = decay * attn_old + lr * (|residual_i|/max_i(|residual_i|))
        """
        attention_old = weights
        r_abs     = jnp.abs(residuals)
        r_max     = jnp.max(r_abs)
        attention_new = hyperparams["gamma_decay"]*attention_old + hyperparams["eta_lr"]*(r_abs/(r_max+1e-12))
        return attention_new