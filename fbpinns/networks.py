"""
Defines standard neural network models

Each network class must inherit from the Network base class.
Each network class must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import jax.numpy as jnp
from jax import random
import jax
from functools import partial


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


class ChebyshevKAN(Network):
    "Chebyshev polynomials"
    @staticmethod
    def init_params(key, in_dim, out_dim, degree, kind=2):
        std = 1.0 / (in_dim * (degree + 1))
        coeffs = std * jax.random.normal(key, (in_dim, out_dim, degree+1))
        assert kind in (1, 2)
        return {"kind": kind}, {"coeffs": coeffs}

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
    

class StackedChebyshevKAN(Network):
    """
    Stacked Chebyshev KAN with arbitrary number of blocks.

    Args:
      dims: List of dimensions [I0, I1, ..., IB] for B blocks.
      degrees: List of polynomial degrees [D0, D1, ..., D_{B-1}] for each block.
      kinds: Optional int or list of ints specifying Chebyshev kind (1 or 2) per block.
    """
    @staticmethod
    def init_params(key, dims, degrees, kinds=2):
        B = len(degrees)
        kinds_list = [kinds]*B if isinstance(kinds, int) else list(kinds)
        keys = jax.random.split(key, B)
        coeffs_list = []
        for i, (d_in, d_out, deg, knd) in enumerate(zip(dims, dims[1:], degrees, kinds_list)):
            static, train = ChebyshevKAN.init_params(keys[i], d_in, d_out, deg, knd)
            coeffs_list.append(train["coeffs"])
        return {"kinds": tuple(kinds_list)}, {"coeffs_list": tuple(coeffs_list)}

    @staticmethod
    def network_fn(all_params, x):
        coeffs_list = all_params["trainable"]["network"]["subdomain"]["coeffs_list"]
        kinds_list = all_params["static"]["network"]["subdomain"]["kinds"]
        # Sequentially apply each Chebyshev block
        y = x
        for coeffs, kind in zip(coeffs_list, kinds_list):
            y = ChebyshevKAN.forward(coeffs, kind, y)
        return y


class OptimizedChebyshevKAN(ChebyshevKAN):

    @staticmethod
    def network_fn(all_params, x):
        # Pull static ints out at Python level
        kind = all_params["static"]["network"]["subdomain"]["kind"]
        coeffs = all_params["trainable"]["network"]["subdomain"]["coeffs"]

        # JIT only over (coeffs, x), capturing kind & degree
        @jax.jit
        def _apply(c, xx):
            return OptimizedChebyshevKAN.forward(c, kind, xx)

        return _apply(coeffs, x)

    @staticmethod
    def forward(coeffs: jnp.ndarray, kind, x: jnp.ndarray) -> jnp.ndarray:
        was_vector = (x.ndim == 1)
        if was_vector:
            x = x[None, :]

        z  = jnp.tanh(x)
        U0 = jnp.ones_like(z)
        U1 = kind * z
        degree = coeffs.shape[-1] - 1
        
        def step(carry, _):
            Um2, Um1 = carry
            Un = 2 * z * Um1 - Um2
            return (Um1, Un), Un

        (_, _), rest = jax.lax.scan(step, (U0, U1), None, length=degree-1)
        rest = jnp.moveaxis(rest, 0, -1)   # rest: (degree-1, batch, in_dim) → (batch, in_dim, degree-1)
        cheb = jnp.concatenate([U0[...,None], U1[...,None], rest], axis=-1)  # → (batch, in_dim, degree+1)
        y = jnp.tensordot(cheb, coeffs, axes=([1,2], [0,2]))  # → (batch, out_dim)
        return y[0] if was_vector else y


class OptimizedStackedChebyshevKAN(StackedChebyshevKAN):

    @staticmethod
    def network_fn(all_params, x):
        kinds = all_params["static"]["network"]["subdomain"]["kinds"]
        coeffs_list = all_params["trainable"]["network"]["subdomain"]["coeffs_list"]

        @jax.jit
        def _apply(coeffs_list_inner, x_inner):
            y = x_inner
            for C, knd in zip(coeffs_list_inner, kinds):
                y = OptimizedChebyshevKAN.forward(C, knd, y)
            return y

        return _apply(coeffs_list, x)


class ChebyshevAdaptiveKAN(Network):
    "Chebyshev polynomials with adaptive activation functions"

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
    

class StackedLegendreKAN(Network):
    """
    Stacked Legendre KAN with arbitrary number of blocks.

    Args:
      dims: List of dimensions [I0, I1, ..., IB] for B blocks.
      degrees: List of polynomial degrees [D0, D1, ..., D_{B-1}] for each block.
    """

    @staticmethod
    def init_params(key, dims, degrees):
        B = len(degrees)
        if len(dims) != B + 1:
            raise ValueError(f"len(dims) must be len(degrees)+1, got dims={len(dims)}, degrees={B}")
        # Split PRNG for each block
        keys = random.split(key, B)
        # Static params: store kinds for each block
        static_params = {}
        # Trainable params: list of coeffs arrays
        coeffs_list = []
        for i in range(B):
            k = keys[i]
            in_dim = dims[i]
            out_dim = dims[i+1]
            deg = degrees[i]
            # ChebyshevKAN returns (static, trainable)
            block_static, block_train = LegendreKAN.init_params(
                k, in_dim, out_dim, degree=deg
            )
            # block_train = {"coeffs": ...}
            coeffs_list.append(block_train["coeffs"])
        trainable_params = {"coeffs_list": tuple(coeffs_list)}
        return static_params, trainable_params

    @staticmethod
    def network_fn(all_params, x):
        coeffs_list = all_params["trainable"]["network"]["subdomain"]["coeffs_list"]
        # Sequentially apply each Chebyshev block
        y = x
        for coeffs in coeffs_list:
            y = LegendreKAN.forward(coeffs, y)
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
    
class MixedBasisKAN(Network):
    """
    Mixture-of-basis KAN combining Chebyshev and Legendre polynomials with adaptive gating.
    """
    @staticmethod
    def init_params(key, input_dim, output_dim, degree):
        # Chebyshev coefficients
        key, subkey = random.split(key)
        coeffs_cheb = ChebyshevKAN.init_params(subkey, input_dim, output_dim, degree)[1]["coeffs"]
        # Legendre coefficients
        key, subkey = random.split(key)
        coeffs_leg  = LegendreKAN.init_params(subkey, input_dim, output_dim, degree)[1]["coeffs"]
        # Gating MLP params
        key, subkey = random.split(key)
        w1 = random.normal(subkey, (input_dim, 16)); b1 = jnp.zeros((16,))
        key, subkey = random.split(key)
        w2 = random.normal(subkey, (16, 2));      b2 = jnp.zeros((2,))
        gating = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
        return {}, {"coeffs_cheb": coeffs_cheb, "coeffs_leg": coeffs_leg, "gating": gating}

    @staticmethod
    def network_fn(all_params, x):
        coeffs_cheb = all_params["trainable"]["coeffs_cheb"]
        coeffs_leg  = all_params["trainable"]["coeffs_leg"]
        gating      = all_params["trainable"]["gating"]

        @jax.jit
        def _apply(cc, cl, gp, xx):
            return MixedBasisKAN.forward(cc, cl, gp, xx)
        return _apply(coeffs_cheb, coeffs_leg, gating, x)

    @staticmethod
    def forward(coeffs_cheb, coeffs_leg, gating, x):
        was_vector = (x.ndim == 1)
        if was_vector:
            x = x[None, :]
        z = jnp.tanh(x)
        # Chebyshev expansion
        U0 = jnp.ones_like(z); U1 = z
        deg_c = coeffs_cheb.shape[-1] - 1
        def cheb_step(carry, _):
            Um2, Um1 = carry; Un = 2*z*Um1 - Um2; return (Um1, Un), Un
        (_, _), rest_c = jax.lax.scan(cheb_step, (U0, U1), None, length=deg_c-1)
        cheb = jnp.concatenate([U0[...,None], U1[...,None], jnp.moveaxis(rest_c,0,-1)], axis=-1)
        y_cheb = jnp.tensordot(cheb, coeffs_cheb, axes=([1,2],[0,2]))
        # Legendre expansion
        P0 = jnp.ones_like(z); P1 = z
        deg_l = coeffs_leg.shape[-1] - 1
        def leg_step(carry, n):
            Pnm2, Pnm1 = carry
            Unp1 = ((2*n+1)*z*Pnm1 - n*Pnm2)/(n+1)
            return (Pnm1, Unp1), Unp1
        (_, _), rest_l = jax.lax.scan(leg_step, (P0,P1), jnp.arange(1,deg_l), length=deg_l-1)
        leg = jnp.concatenate([P0[...,None], P1[...,None], jnp.moveaxis(rest_l,0,-1)], axis=-1)
        y_leg = jnp.tensordot(leg, coeffs_leg, axes=([1,2],[0,2]))
        # Gating
        h = jnp.tanh(jnp.dot(x, gating['w1']) + gating['b1'])
        logits = jnp.dot(h, gating['w2']) + gating['b2']
        alpha = jax.nn.softmax(logits, axis=-1)
        y = alpha[...,0:1]*y_cheb + alpha[...,1:2]*y_leg
        return y[0] if was_vector else y

class StackedMixedBasisKAN(MixedBasisKAN):
    """
    Stacked mixture-of-basis KAN: sequence of MixedBasisKAN blocks.
    """
    @staticmethod
    def init_params(key, dims, degrees):
        B = len(degrees)
        keys = random.split(key, B)
        cheb_list, leg_list, gate_list = [], [], []
        for i,(din,dout,deg) in enumerate(zip(dims[:-1], dims[1:], degrees)):
            subkey = keys[i]
            # init block params
            block_static, block_train = MixedBasisKAN.init_params(subkey, din, dout, deg)
            cheb_list.append(block_train['coeffs_cheb'])
            leg_list.append(block_train['coeffs_leg'])
            gate_list.append(block_train['gating'])
        return {}, {
            'coeffs_cheb_list': tuple(cheb_list),
            'coeffs_leg_list':  tuple(leg_list),
            'gating_list':     tuple(gate_list)
        }

    @staticmethod
    def network_fn(all_params, x):
        cheb_list = all_params['trainable']['coeffs_cheb_list']
        leg_list  = all_params['trainable']['coeffs_leg_list']
        gate_list = all_params['trainable']['gating_list']
        @jax.jit
        def _apply(ccs, cls, gs, xx):
            y = xx
            for c_cheb, c_leg, g in zip(ccs, cls, gs):
                y = MixedBasisKAN.forward(c_cheb, c_leg, g, y)
            return y
        return _apply(cheb_list, leg_list, gate_list, x)


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

# Spline-based KAN implementation in JAX
class SplineKAN(Network):
    """Spline-based Kolmogorov-Arnold Network (KAN)."""

    @staticmethod
    def init_params(key, in_dim, out_dim, grid_size=5, spline_order=3,
                    scale_base=1.0, scale_spline=1.0, grid_range=[-1, 1]):
        """Initialise class parameters.

        Args:
            key: JAX random key.
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            grid_size (int): Number of grid intervals.
            spline_order (int): Order of the B-spline. k=3 is cubic.
            scale_base (float): Scale for initializing the base weights.
            scale_spline (float): Scale for initializing the spline weights.
            grid_range (list): The range of the grid, e.g., [-1, 1].

        Returns:
            A tuple of dictionaries for static and trainable parameters.
        """
        # --- Static Parameters ---
        # These are fixed after initialization.
        h = (grid_range[1] - grid_range[0]) / grid_size
        # Extend the grid to handle boundary conditions for splines
        grid = jnp.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        grid = jnp.tile(grid, (in_dim, 1))

        static_params = {
            "grid": grid,
            "spline_order": spline_order,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "grid_size": grid_size
        }

        # --- Trainable Parameters ---
        # These are the parameters that will be learned during training.
        key, base_key, spline_key, scaler_key = jax.random.split(key, 4)

        # Base weights (similar to a standard linear layer)
        # Using Kaiming uniform initialization
        lim_base = scale_base * jnp.sqrt(1.0 / in_dim)
        base_weight = jax.random.uniform(base_key, (out_dim, in_dim), minval=-lim_base, maxval=lim_base)
        
        # Spline weights (coefficients for the B-spline basis functions)
        lim_spline = scale_spline * jnp.sqrt(1.0 / in_dim)
        spline_weight = jax.random.uniform(
            spline_key,
            (out_dim, in_dim, grid_size + spline_order),
            minval=-lim_spline,
            maxval=lim_spline
        )
        
        # Learnable scaler for the spline component
        spline_scaler = jax.random.uniform(
            scaler_key,
            (out_dim, in_dim),
            minval=-lim_base,
            maxval=lim_base
        )
        
        trainable_params = {
            "base_weight": base_weight,
            "spline_weight": spline_weight,
            "spline_scaler": spline_scaler,
        }
        
        return static_params, trainable_params

    # @staticmethod
    # def _b_splines(x, grid, spline_order):
    #     """
    #     spline_order is a Python int (static), so we can write
    #     a normal Python loop and let JAX unroll it.
    #     """
    #     x = jnp.atleast_2d(x)[..., None]    # (batch, in_dim, 1)
    #     # degree‐0 basis
    #     bases = jnp.where((x >= grid[:, :-1]) & (x < grid[:, 1:]), 1.0, 0.0)

    #     # now a pure Python loop over a real int
    #     for k in range(1, 3 + 1): #spline_order
    #         left_num   = x - grid[:, : -(k+1)]
    #         left_den   = grid[:, k:-1] - grid[:, : -(k+1)]
    #         right_num  = grid[:, k+1:] - x
    #         right_den  = grid[:, k+1:] - grid[:, 1:-k]

    #         # avoid division by zero
    #         left_den   = jnp.where(left_den   == 0, 1.0, left_den)
    #         right_den  = jnp.where(right_den  == 0, 1.0, right_den)

    #         term1 = (left_num  / left_den)[..., :-1] * bases[..., :-1]
    #         term2 = (right_num / right_den)[..., 1:]  * bases[..., 1:]
    #         bases = term1 + term2

    #     return bases

    def _b_splines_cubic(x, grid):
        """
        Cubic (order=3) B-spline basis builder, fully unrolled.
        Returns shape (batch, in_dim, grid_size + 3).
        """
        # ensure a batch dim, and lift features to the last axis
        x = jnp.atleast_2d(x)
        b = x[..., None]     # (batch, in_dim, 1)

        # --- k = 0: piecewise-constant ---
        deg0 = jnp.where((b >= grid[:, :-1]) & (b < grid[:, 1:]), 1.0, 0.0)
        # deg0.shape == (batch, in_dim, grid_len - 1)

        # --- k = 1 ---
        # intervals lengths
        # grid_len = grid.shape[1]
        # num0 = grid_len - 1
        left1  = (b - grid[:, : -2])[..., :-1] / (grid[:, 1:-1] - grid[:, : -2])[..., :-1]  * deg0[..., :-1]
        right1 = (grid[:, 2: ] - b)[..., 1: ] / (grid[:, 2: ] - grid[:, 1:-1])[..., 1: ] * deg0[..., 1: ]
        deg1 = left1 + right1
        # deg1.shape == (batch, in_dim, grid_len - 2)

        # --- k = 2 ---
        left2  = (b - grid[:, : -3])[..., :-1] / (grid[:, 2:-1] - grid[:, : -3])[..., :-1]  * deg1[..., :-1]
        right2 = (grid[:, 3: ] - b)[..., 1: ] / (grid[:, 3: ] - grid[:, 1:-2])[..., 1: ] * deg1[..., 1: ]
        deg2 = left2 + right2
        # deg2.shape == (batch, in_dim, grid_len - 3)

        # --- k = 3 ---
        left3  = (b - grid[:, : -4])[..., :-1] / (grid[:, 3:-1] - grid[:, : -4])[..., :-1]  * deg2[..., :-1]
        right3 = (grid[:, 4: ] - b)[..., 1: ] / (grid[:, 4: ] - grid[:, 1:-3])[..., 1: ] * deg2[..., 1: ]
        deg3 = left3 + right3
        # deg3.shape == (batch, in_dim, grid_len - 4) == (batch, in_dim, grid_size + 3)

        return deg3

    @staticmethod
    def network_fn(all_params, x):
        """The forward pass of the Spline KAN layer.
        
        Args:
            all_params (dict): A dictionary containing 'static' and 'trainable' parameter dicts.
            x (jnp.array): Input data. Shape (batch, in_dim) or (in_dim,).
            
        Returns:
            jnp.array: The output of the KAN layer.
        """
        # Unpack parameters
        static_p = all_params["static"]["network"]["subdomain"]
        trainable_p = all_params["trainable"]["network"]["subdomain"]
        
        # The main forward computation
        return SplineKAN.forward(x, static_p, trainable_p)

    @staticmethod
    def forward(x, static_params, trainable_params):
        """The core computation of the forward pass."""
        # Unpack parameters
        grid = static_params["grid"]
        spline_order = static_params["spline_order"]
        base_weight = trainable_params["base_weight"]
        spline_weight = trainable_params["spline_weight"]
        spline_scaler = trainable_params["spline_scaler"]
        
        # Ensure input has a batch dimension
        original_shape = x.shape
        x = jnp.atleast_2d(x)
        
        # 1. Base component (residual-like connection)
        # We use SiLU (Swish) activation, same as the PyTorch example's default
        base_activation = jax.nn.silu(x)
        base_output = base_activation @ base_weight.T
        
        # 2. Spline component
        # Compute B-spline basis functions
        print(spline_order.shape)
        # bases = SplineKAN._b_splines(x, grid, spline_order)
        bases = SplineKAN._b_splines_cubic(x, grid)
        
        # Scale the spline weights
        scaled_spline_weight = spline_weight * spline_scaler[..., None]
        
        # Compute spline output via einsum for efficient batch processing
        spline_output = jnp.einsum('bid,oid->bo', bases, scaled_spline_weight)
        
        # Total output is the sum of the two components
        output = base_output + spline_output
        
        # Return result with the original batch dimension (or lack thereof)
        if len(original_shape) == 1:
            return output[0]
        return output
    
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
