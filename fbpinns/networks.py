"""
Defines standard neural network models

Each network class must inherit from the Network base class.
Each network class must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import jax.numpy as jnp
from jax import random
import jax


class Network:
    """Base neural network class to be inherited by different neural network classes.

    Note all methods in this class are jit compiled / used by JAX,
    so they must not include any side-effects!
    (A side-effect is any effect of a function that doesn’t appear in its output)
    This is why only static methods are defined.
    """

    # required methods

    @staticmethod
    def init_params(*args):
        """Initialise class parameters.
        Returns tuple of dicts ({k: pytree}, {k: pytree}) containing static and trainable parameters"""
        raise NotImplementedError

    @staticmethod
    def network_fn(params, x):
        """Forward model, for a SINGLE point with shape (xd,)"""
        raise NotImplementedError

class ChebyshevAdaptiveKAN(Network):
    "Chebyshev polynomials"

    @staticmethod
    def init_params(key, input_dim, output_dim, degree, kind=2):
        mean = 0
        std = 1/(input_dim * (degree + 1))
        coeffs = mean + std * random.normal(key, (input_dim, output_dim, degree+1))
        a = jnp.ones((degree+1))
        assert kind in {1, 2}
        trainable_params = {"coeffs": coeffs, "a": a}
        return {"kind": kind}, trainable_params

    @staticmethod
    def network_fn(all_params, x):
        coeffs = all_params["trainable"]["network"]["subdomain"]["coeffs"]
        kind = all_params["static"]["network"]["subdomain"]["kind"]
        a = all_params["trainable"]["network"]["subdomain"]["a"]
        return ChebyshevAdaptiveKAN.forward(coeffs, kind, a, x)

    @staticmethod
    def forward(coeffs, kind, a, x):
        input_dim = coeffs.shape[0]
        degree = coeffs.shape[-1] - 1

        batch_size = x.shape[0]

        cheb = jnp.ones((batch_size, input_dim, degree + 1))
        if degree >= 1:
            # initialisation based on first or second polynomial kind
            xa = jnp.tanh(x/a[0]) # a[0] * 
            cheb = cheb.at[:, :, 1].set(kind * xa)
        for d in range(2, degree + 1):
            xa = jnp.tanh(x/a[d]) # a[d] * 
            cheb = cheb.at[:, :, d].set(2 * xa * cheb[:, :, d - 1] - cheb[:, :, d - 2])

        y = jnp.einsum("bid,iod->bo", cheb, coeffs)
        y = y if len(x.shape) > 1 else y[0]
        return y
    
class StackedChebyshevKAN(Network):
    """Two ChebyshevKAN blocks in series"""

    @staticmethod
    def init_params(key, input_dim, hidden_dim, output_dim, degree, kind=2):
        # split key for two blocks and skip
        k1, k2, k3 = random.split(key, 3)
        # block 1: input_dim -> hidden_dim
        static1, params1 = ChebyshevKAN.init_params(k1, input_dim, hidden_dim, degree, kind)
        # block 2: hidden_dim -> output_dim
        static2, params2 = ChebyshevKAN.init_params(k2, hidden_dim, output_dim, degree, kind)
        # linear skip: input_dim -> output_dim
        w_skip = random.normal(k3, (input_dim, output_dim)) * jnp.sqrt(2 / input_dim)
        b_skip = jnp.zeros((output_dim,))

        static = {"kind1": static1["kind"], "kind2": static2["kind"]}
        trainable = {"block1": params1["coeffs"], "block2": params2["coeffs"], "w_skip": w_skip, "b_skip": b_skip}
        return static, trainable

    @staticmethod
    def network_fn(all_params, x):
        # unpack
        kind1 = all_params["static"]["network"]["subdomain"]["kind1"]
        kind2 = all_params["static"]["network"]["subdomain"]["kind2"]
        coeffs1 = all_params["trainable"]["network"]["subdomain"]["block1"]
        coeffs2 = all_params["trainable"]["network"]["subdomain"]["block2"]
        w_skip = all_params["trainable"]["network"]["subdomain"]["w_skip"]
        b_skip = all_params["trainable"]["network"]["subdomain"]["b_skip"]

        # first KAN block
        y1 = ChebyshevKAN.forward(coeffs1, kind1, x)
        # second KAN block
        y2 = ChebyshevKAN.forward(coeffs2, kind2, y1)
        # skip connection
        # skip = x @ w_skip + b_skip
        return y2
        # return skip + y2


    
class ChebyshevKAN(Network):
    "Chebyshev polynomials"

    @staticmethod
    def init_params(key, input_dim, output_dim, degree, kind=2):
        mean = 0
        std = 1/(input_dim * (degree + 1))
        coeffs = mean + std * random.normal(key, (input_dim, output_dim, degree+1))
        assert kind in {1, 2}
        trainable_params = {"coeffs": coeffs}
        return {"kind": kind}, trainable_params

    @staticmethod
    def network_fn(all_params, x):
        coeffs = all_params["trainable"]["network"]["subdomain"]["coeffs"]
        kind = all_params["static"]["network"]["subdomain"]["kind"]
        return ChebyshevKAN.forward(coeffs, kind, x)

    @staticmethod
    def forward(coeffs, kind, x):
        input_dim = coeffs.shape[0]
        degree = coeffs.shape[-1] - 1

        x = jnp.tanh(x)
        batch_size = x.shape[0]

        cheb = jnp.ones((batch_size, input_dim, degree + 1))
        if degree >= 1:
            # initialisation based on first or second polynomial kind
            cheb = cheb.at[:, :, 1].set(kind * x)
        for d in range(2, degree + 1):
            cheb = cheb.at[:, :, d].set(2 * x * cheb[:, :, d - 1] - cheb[:, :, d - 2])

        y = jnp.einsum("bid,iod->bo", cheb, coeffs)
        y = y if len(x.shape) > 1 else y[0]
        return y
    
class LegendreKAN(Network):
    "Legendre polynomials"

    @staticmethod
    def init_params(key, input_dim, output_dim, degree):
        mean = 0
        std = 1/(input_dim * (degree + 1))
        coeffs = mean + std * random.normal(key, (input_dim, output_dim, degree+1))
        trainable_params = {"coeffs": coeffs}
        return {}, trainable_params

    @staticmethod
    def network_fn(all_params, x):
        coeffs = all_params["trainable"]["network"]["subdomain"]["coeffs"]
        return LegendreKAN.forward(coeffs, x)

    @staticmethod
    def forward(coeffs, x):
        input_dim = coeffs.shape[0]
        degree = coeffs.shape[-1] - 1

        x = jnp.tanh(x)
        batch_size = x.shape[0]

        cheb = jnp.ones((batch_size, input_dim, degree + 1))
        if degree >= 1:
            cheb = cheb.at[:, :, 1].set(x)
        for d in range(2, degree + 1):
            cheb = cheb.at[:, :, d].set( ((2 * (d-1) + 1) / (d)) * x * cheb[:, :, d - 1] - ((d-1) / (d)) * cheb[:, :, d - 2])

        y = jnp.einsum("bid,iod->bo", cheb, coeffs)
        y = y if len(x.shape) > 1 else y[0]
        return y

class LegendreAdaptiveKAN(Network):

    @staticmethod
    def init_params(key, input_dim, output_dim, degree):
        mean = 0
        std = 1/(input_dim * (degree + 1))
        coeffs = mean + std * random.normal(key, (input_dim, output_dim, degree+1))
        a = jnp.ones((degree+1))
        trainable_params = {"coeffs": coeffs, "a": a}
        return {}, trainable_params

    @staticmethod
    def network_fn(all_params, x):
        coeffs = all_params["trainable"]["network"]["subdomain"]["coeffs"]
        a = all_params["trainable"]["network"]["subdomain"]["a"]
        return LegendreAdaptiveKAN.forward(coeffs, a, x)

    @staticmethod
    def forward(coeffs, a, x):
        input_dim = coeffs.shape[0]
        degree = coeffs.shape[-1] - 1

        batch_size = x.shape[0]

        cheb = jnp.ones((batch_size, input_dim, degree + 1))
        if degree >= 1:
            xa = jnp.tanh(x/a[0])
            cheb = cheb.at[:, :, 1].set(xa)
        for d in range(2, degree + 1):
            xa = jnp.tanh(x/a[d])
            cheb = cheb.at[:, :, d].set( ((2 * (d-1) + 1) / (d)) * xa * cheb[:, :, d - 1] - ((d-1) / (d)) * cheb[:, :, d - 2])

        y = jnp.einsum("bid,iod->bo", cheb, coeffs)
        y = y if len(x.shape) > 1 else y[0]
        return y
    
class HermiteKAN(Network):
    "Hermite polynomials"

    @staticmethod
    def init_params(key, input_dim, output_dim, degree):
        # Initialize the Hermite coefficients with mean=0, std=1/(input_dim*(degree+1))
        # mean = 0.0
        # std = 1.0 / (input_dim * (degree + 1))
        # coeffs = mean + std * random.normal(key, (input_dim, output_dim, degree + 1))
        # 2) Xavier‐uniform for the linear term H1(x)=2x
        #    so that the initial map is a well‐scaled Dense layer
        coeffs = jnp.zeros((input_dim, output_dim, degree + 1))
        fan_in, fan_out = input_dim, output_dim
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        key, subkey = random.split(key)
        linear_init = random.uniform(subkey,
                                     (input_dim, output_dim),
                                     minval=-limit,
                                     maxval=limit)

        coeffs = coeffs.at[:, :, 1].set(linear_init)
        return {}, {"coeffs": coeffs}

    @staticmethod
    def network_fn(all_params, x):
        coeffs = all_params["trainable"]["network"]["subdomain"]["coeffs"]
        return HermiteKAN.forward(coeffs, x)

    @staticmethod
    def forward(coeffs, x):
        input_dim = coeffs.shape[0]
        degree = coeffs.shape[-1] - 1

        x = jnp.tanh(x)
        batch_size = x.shape[0]

        # Build Hermite basis: shape [batch, input_dim, degree+1]
        herm = jnp.ones((batch_size, input_dim, degree + 1))
        print(herm.shape)
        if degree >= 1:
            # H₁(x) = 2 x
            herm = herm.at[:, :, 1].set(2 * x)

        # Recurrence: Hₙ(x) = 2 x Hₙ₋₁(x) − 2(n−1) Hₙ₋₂(x)
        for n in range(2, degree + 1):
            herm = herm.at[:, :, n].set(
                2 * x * herm[:, :, n - 1] - 2 * (n - 1) * herm[:, :, n - 2]
            )

        # einsum over batch, input_dim, degree to get [batch_size, output_dim]
        y = jnp.einsum("bid,iod->bo", herm, coeffs)

        # If original x was 1D, drop batch dim
        return y if x.ndim > 1 else y[0]

class FCN(Network):
    "Fully connected network"

    @staticmethod
    def init_params(key, layer_sizes):
        keys = random.split(key, len(layer_sizes)-1)
        params = [FCN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params 

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(1/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        return w,b

    @staticmethod
    def network_fn(all_params, x):
        params = all_params["trainable"]["network"]["subdomain"]["layers"]
        return FCN.forward(params, x)

    @staticmethod
    def forward(params, x):
        for w, b in params[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.tanh(x)
        w, b = params[-1]
        x = jnp.dot(w, x) + b
        return x

class AdaptiveFCN(Network):
    "Fully connected network with adaptive activations"

    @staticmethod
    def init_params(key, layer_sizes):
        keys = random.split(key, len(layer_sizes)-1)
        params = [AdaptiveFCN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(1/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        a = jnp.ones_like(b)
        return w,b,a

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w, b, a in params[:-1]:
            x = jnp.dot(w, x) + b
            x = a*jnp.tanh(x/a)
        w, b, _ = params[-1]
        x = jnp.dot(w, x) + b
        return x

class SIREN(Network):
    "Fully connected network with sin activations"

    @staticmethod
    def init_params(key, layer_sizes):
        keys = random.split(key, len(layer_sizes)-1)
        params = [SIREN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(6/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        return w,b

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w, b in params[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.sin(x)
        w, b = params[-1]
        x = jnp.dot(w, x) + b
        return x

class AdaptiveSIREN(Network):
    "Fully connected network with adaptive sin activations"

    @staticmethod
    def init_params(key, layer_sizes):
        keys = random.split(key, len(layer_sizes)-1)
        params = [AdaptiveSIREN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(6/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        c,o = jnp.ones_like(b), jnp.ones_like(b)
        return w,b,c,o

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w,b,c,o in params[:-1]:
            x = jnp.dot(w, x) + b
            x = c*jnp.sin(o*x)
        w,b,_,_ = params[-1]
        x = jnp.dot(w, x) + b
        return x

class FourierFCN(FCN):
    "Fully connected network with Fourier features"

    @staticmethod
    def init_params(key, layer_sizes, mu, sd, n_features):

        # get Fourier feature parameters
        key, subkey = random.split(key)
        omega = 2*jnp.pi*(mu+sd*random.normal(subkey, (n_features, layer_sizes[0])))
        layer_sizes = [2*n_features]+list(layer_sizes)[1:]

        # get FCN parameters
        keys = random.split(key, len(layer_sizes)-1)
        params = [FCN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {"omega":omega}, trainable_params

    @staticmethod
    def network_fn(params, x):
        omega = params["static"]["network"]["subdomain"]["omega"]
        params = params["trainable"]["network"]["subdomain"]["layers"]
        x = jnp.dot(omega, x)
        x = jnp.concatenate([jnp.sin(x), jnp.cos(x)])# (2*n_features)
        for w, b in params[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.tanh(x)
        w, b = params[-1]
        x = jnp.dot(w, x) + b
        return x


def norm(mu, sd, x):
    return (x-mu)/sd

def unnorm(mu, sd, x):
    return x*sd + mu



if __name__ == "__main__":

    x = jnp.ones(2)
    key = random.PRNGKey(0)
    layer_sizes = [2,16,32,16,1]
    for NN in [FCN, AdaptiveFCN, SIREN, AdaptiveSIREN, FourierFCN]:
        network = NN
        if NN is FourierFCN:
            ps_ = network.init_params(key, layer_sizes, 0, 1, 10)
        else:
            ps_ = network.init_params(key, layer_sizes)
        params = {"static":{"network":{"subdomain":ps_[0]}},
                  "trainable":{"network":{"subdomain":ps_[1]}}}
        print(x.shape, network.network_fn(params, x).shape, NN.__name__)


# class ChebyshevResKAN(Network):
#     "Chebyshev polynomials with a residual connection from x via a learned projection."

#     @staticmethod
#     def init_params(key, input_dim, output_dim, degree, kind=2):
#         # Initialize coefficients for the Chebyshev basis as before.
#         mean = 0.0
#         std = 1/(input_dim * (degree + 1))
#         coeffs = mean + std * random.normal(key, (input_dim, output_dim, degree + 1))
#         # Additional parameters for the residual branch that projects x from input_dim to output_dim.
#         res_key, key = random.split(key)
#         W_res = random.normal(res_key, (input_dim, output_dim)) * (1.0 / input_dim)
#         b_res = jnp.zeros((output_dim,))
#         trainable_params = {
#             "coeffs": coeffs,
#             "W_res": W_res,
#             "b_res": b_res,
#         }
#         # "kind" is treated as a static parameter.
#         return {"kind": kind}, trainable_params

#     @staticmethod
#     def network_fn(all_params, x):
#         coeffs = all_params["trainable"]["network"]["subdomain"]["coeffs"]
#         W_res = all_params["trainable"]["network"]["subdomain"]["W_res"]
#         b_res = all_params["trainable"]["network"]["subdomain"]["b_res"]
#         kind = all_params["static"]["network"]["subdomain"]["kind"]
#         poly_out = ChebyshevKAN.forward(coeffs, kind, x)
#         # Compute the residual projection: shape (batch_size, output_dim)
#         res_proj = jnp.tanh(jnp.dot(x, W_res) + b_res)
#         # Add the residual to the polynomial output.
#         # Note: the transformation applied to x in the polynomial branch is tanh(x),
#         # which we can also mimic in the residual branch (or not) depending on what works best.
#         return poly_out + res_proj

#     @staticmethod
#     def forward(coeffs, kind, x):
#         input_dim = coeffs.shape[0]
#         degree = coeffs.shape[-1] - 1

#         # Apply a non-linear squashing function to x (as in the original code) for the polynomial branch.
#         # Note: The residual branch in our example uses the raw x; but alternatively, one could use a non-linear version.
#         x_poly = jnp.tanh(x)
#         batch_size = x_poly.shape[0]

#         # Initialize the Chebyshev basis: shape (batch_size, input_dim, degree+1)
#         basis = jnp.ones((batch_size, input_dim, degree + 1))
#         if degree >= 1:
#             basis = basis.at[:, :, 1].set(kind * x_poly)
#         for d in range(2, degree + 1):
#             basis = basis.at[:, :, d].set(2 * x_poly * basis[:, :, d - 1] - basis[:, :, d - 2])
            
#         # Compute the polynomial output using Einstein summation.
#         poly_out = jnp.einsum("bid,iod->bo", basis, coeffs)
#         return poly_out
