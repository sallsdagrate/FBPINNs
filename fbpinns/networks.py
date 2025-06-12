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
from typing import List, Dict, Tuple


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


class OrthogonalPolynomialKAN(Network):
    """
    General JIT‐compiled “Orthogonal Polynomial KAN” for any 3‐term recurrence.

    Recurrence form (for n = 2..D):
        P₀(x) = 1,
        P₁(x) = (a₁ x + b₁)/d₁,
        Pₙ(x) = [ (aₙ x + bₙ)·P_{n-1}(x)  –  cₙ·P_{n-2}(x ) ] / dₙ,    n ≥ 2.

    The user supplies four 1D arrays (each length D+1):
        a = [a₀, a₁, …, a_D],
        b = [b₀, b₁, …, b_D],
        c = [c₀, c₁, …, c_D],
        d = [d₀, d₁, …, d_D].

    Typically a₀=0, b₀=0, c₁=0, d₀=1 so that P₀=1, P₁=(a₁x + b₁)/d₁.
    """

    @staticmethod
    def init_params(
        key,
        in_dim: int,
        out_dim: int,
        degree: int,
        recurrence_coefs: Dict[str, jnp.ndarray]
    ) -> Tuple[Dict, Dict]:
        """
        - `degree` = D.  Trainable coeffs shape = (in_dim, out_dim, D+1).
        - `recurrence_coefs` must be a dict with keys "a","b","c","d",
          each a jnp.ndarray of shape (D+1,).
        """

        std = 1.0 / (in_dim * (degree + 1))
        coeffs = std * jax.random.normal(key, (in_dim, out_dim, degree + 1))

        static_params = {
            "recurrence": {
                "a": recurrence_coefs["a"],
                "b": recurrence_coefs["b"],
                "c": recurrence_coefs["c"],
                "d": recurrence_coefs["d"]
            }
        }
        trainable_params = {
            "coeffs": coeffs
        }
        return static_params, trainable_params

    @staticmethod
    def network_fn(all_params, x: jnp.ndarray) -> jnp.ndarray:
        """
        Pull out:
          - coeffs (trainable):   shape = (in_dim, out_dim, D+1)
          - a, b, c, d (static):  each shape = (D+1,)

        JIT‐compile a single forward over (coeffs, a, b, c, d, x).
        """
        coeffs = all_params["trainable"]["network"]["coeffs"]
        rec   = all_params["static"]["network"]["recurrence"]
        a     = rec["a"]
        b     = rec["b"]
        c     = rec["c"]
        d     = rec["d"]

        @jax.jit
        def _apply_jit(coeffs_jit, a_jit, b_jit, c_jit, d_jit, x_inp) -> jnp.ndarray:
            return OrthogonalPolynomialKAN._forward_jit(
                coeffs_jit, a_jit, b_jit, c_jit, d_jit, x_inp
            )

        return _apply_jit(coeffs, a, b, c, d, x)

    @staticmethod
    def _forward_jit(
        coeffs, a, b, c, d, x
    ) -> jnp.ndarray:
        """
        Build {P₀…P_D}( tanh(x) ) via a single lax.scan, then contract.

        - coeffs: (in_dim, out_dim, D+1)
        - a,b,c,d: each (D+1,)
        - x: (batch_size, in_dim) or (in_dim,) 
        """
        was_vector = (x.ndim == 1)
        x_batch = x if not was_vector else x[None, :]    # → (batch_size, in_dim)
        # z = jnp.tanh(x_batch)                            # (batch_size, in_dim)
        z = x
        batch_size, in_dim = z.shape
        D = coeffs.shape[-1] - 1

        # P₀(x) = 1
        P0 = jnp.ones_like(z)                            # (batch_size, in_dim)

        if D == 0:
            basis = P0[..., None]                        # (batch_size, in_dim, 1)
            y = jnp.tensordot(basis, coeffs, axes=([1, 2], [0, 2]))
            return y[0] if was_vector else y

        # P₁(x) = (a₁·x + b₁)/d₁
        P1 = (a[1] * z + b[1]) / d[1]                     # (batch_size, in_dim)

        if D == 1:
            basis = jnp.stack([P0, P1], axis=-1)         # (batch_size, in_dim, 2)
            y = jnp.tensordot(basis, coeffs, axes=([1, 2], [0, 2]))
            return y[0] if was_vector else y

        # For n=2..D: Pₙ = [ (aₙ x + bₙ)·P_{n-1} – cₙ·P_{n-2} ] / dₙ
        def step_fn(carry, n_idx):
            prev2, prev1 = carry                        # each (batch_size, in_dim)
            term1 = (a[n_idx] * z + b[n_idx]) * prev1   # (batch_size, in_dim)
            term2 = c[n_idx] * prev2                    # (batch_size, in_dim)
            Pn    = (term1 - term2) / d[n_idx]          # (batch_size, in_dim)
            return (prev1, Pn), Pn

        ns = jnp.arange(2, D + 1)
        init_carry = (P0, P1)
        (_, _), scanned = jax.lax.scan(step_fn, init_carry, ns)
        # scanned: (D-1, batch_size, in_dim) → moveaxis → (batch_size, in_dim, D-1)
        scanned = jnp.moveaxis(scanned, 0, -1)

        basis = jnp.concatenate(
            [P0[..., None], P1[..., None], scanned],
            axis=-1
        )  # (batch_size, in_dim, D+1)

        y = jnp.tensordot(basis, coeffs, axes=([1, 2], [0, 2]))  # (batch_size, out_dim)
        return y[0] if was_vector else y


class StackedOrthogonalPolynomialKAN(Network):
    """
    “Stacked” version: apply B blocks in series, each with its own 3‐term recurrence.

    Args:
      dims:        [I₀, I₁, …, I_B], length = B+1
      degrees:     [D₀, D₁, …, D_{B-1}], length = B
      recurrences: List of length B, where
          recurrences[i] = {
              "a": jnp.ndarray(shape=(D_i+1,)),
              "b": jnp.ndarray(shape=(D_i+1,)),
              "c": jnp.ndarray(shape=(D_i+1,)),
              "d": jnp.ndarray(shape=(D_i+1,))
          }
    """

    @staticmethod
    def init_params(
        key ,
        dims: List[int],
        degrees: List[int],
        recurrences: List[Dict[str, jnp.ndarray]]
    ) -> Tuple[Dict, Dict]:
        """
        For each block i=0..B-1:
          - dims[i] → dims[i+1]
          - degree = degrees[i]
          - recurrences[i] has arrays "a","b","c","d" of length degree+1.

        Splits `key` into B sub‐keys, initializes one (in_i, out_i, degree_i+1) trainable
        coeff tensor per block, and stores the recurrence arrays in static.
        """
        B = len(degrees)
        assert len(dims) == B + 1
        assert len(recurrences) == B

        keys = jax.random.split(key, B)
        coeffs_list, a_list, b_list, c_list, d_list = [], [], [], [], []

        for i in range(B):
            d_in, d_out = dims[i], dims[i+1]
            D_i = degrees[i]

            std = 1.0 / (d_in * (D_i + 1))
            coeffs_block = std * jax.random.normal(keys[i], (d_in, d_out, D_i + 1))

            coeffs_list.append(coeffs_block)
            a_list.append(recurrences[i]["a"])
            b_list.append(recurrences[i]["b"])
            c_list.append(recurrences[i]["c"])
            d_list.append(recurrences[i]["d"])

        static_params = {
            "recurrences": {
                "a_list": tuple(a_list),
                "b_list": tuple(b_list),
                "c_list": tuple(c_list),
                "d_list": tuple(d_list),
            }
        }
        trainable_params = {
            "coeffs_list": tuple(coeffs_list)
        }
        return static_params, trainable_params

    @staticmethod
    def network_fn(all_params, x: jnp.ndarray) -> jnp.ndarray:
        """
        JIT‐compile a single function that loops through all B blocks.  Each block i:
          - takes yᶦ (shape=(batch_size, dims[i]))
          - builds basis P₀…P_{D_i} at z = tanh(yᶦ)
          - contracts with coeffs_list[i] (shape=(dims[i], dims[i+1], D_i+1))
          - outputs yⁱ⁺¹ (shape=(batch_size, dims[i+1])).

        Returns final y⁽ᴮ⁾ (shape=(batch_size, dims[B])) or vector if input was vector.
        """
        coeffs_list = all_params["trainable"]["network"]["subdomain"]["coeffs_list"]
        rec = all_params["static"]["network"]["subdomain"]["recurrences"]
        a_list = rec["a_list"]
        b_list = rec["b_list"]
        c_list = rec["c_list"]
        d_list = rec["d_list"]

        @jax.jit
        def _apply_all(coeffs_tuple, a_tuple, b_tuple, c_tuple, d_tuple, x_inp) -> jnp.ndarray:
            y = x_inp
            was_vector = (y.ndim == 1)
            if was_vector:
                y = y[None, :]

            for coeffs_block, a_block, b_block, c_block, d_block in zip(
                coeffs_tuple, a_tuple, b_tuple, c_tuple, d_tuple
            ):
                y = OrthogonalPolynomialKAN._forward_jit(
                    coeffs_block, a_block, b_block, c_block, d_block, y
                )

            return y[0] if was_vector else y

        return _apply_all(coeffs_list, a_list, b_list, c_list, d_list, x)

class StackedLegendreKAN_(Network):
    """
    Wrapper for a stacked Legendre‐polynomial KAN using the original (2n−1) normalization,
    implemented without explicit Python‐level branching. Delegates to StackedOrthogonalPolynomialKAN.
    
    Args:
      dims:    [I₀, I₁, ..., I_B], length = B+1
      degrees: [D₀, D₁, ..., D_{B-1}], length = B
    """

    @staticmethod
    def init_params(
        key ,
        dims: List[int],
        degrees: List[int]
    ) -> Tuple[Dict, Dict]:
        B = len(degrees)
        assert len(dims) == B + 1

        # Split the PRNG key into B sub‐keys
        keys = jax.random.split(key, B)

        recurrences: List[Dict[str, jnp.ndarray]] = []
        for i, D in enumerate(degrees):
            n = jnp.arange(D + 1, dtype=jnp.float32)
            a = jnp.where(n == 0.0,
                          0.0,
                          jnp.where(n == 1.0,
                                    1.0,
                                    2.0 * n - 1.0))
            b = jnp.zeros(D + 1, dtype=jnp.float32)
            c = jnp.where(n >= 2.0, n - 1.0, 0.0)
            d = jnp.where(n >= 2.0, n, 1.0)
            recurrences.append({"a": a, "b": b, "c": c, "d": d})

        return StackedOrthogonalPolynomialKAN.init_params(key, dims, degrees, recurrences)

    @staticmethod
    def network_fn(all_params, x: jnp.ndarray) -> jnp.ndarray:
        return StackedOrthogonalPolynomialKAN.network_fn(all_params, x)


class StackedChebyshevKAN_(Network):
    """
    Wrapper for a stacked Chebyshev‐polynomial KAN (first kind, Tₙ).
    Implemented concisely via vectorized array expressions.
    
    Args:
      dims:    [I₀, I₁, ..., I_B], length = B+1
      degrees: [D₀, D₁, ..., D_{B-1}], length = B
    """

    @staticmethod
    def init_params(
        key ,
        dims: List[int],
        degrees: List[int],
        kind: float = 2.
    ) -> Tuple[Dict, Dict]:
        B = len(degrees)
        assert len(dims) == B + 1

        # Split the PRNG key into B sub‐keys
        keys = jax.random.split(key, B)

        recurrences: List[Dict[str, jnp.ndarray]] = []
        for i, D in enumerate(degrees):
            n = jnp.arange(D + 1, dtype=jnp.float32)
            a = jnp.where(n == 0.0,
                          0.0,
                          jnp.where(n == 1.0,
                                    kind,
                                    2.0))
            b = jnp.zeros(D + 1, dtype=jnp.float32)
            c = jnp.where(n >= 2.0, 1.0, 0.0)
            d = jnp.ones(D + 1, dtype=jnp.float32)
            recurrences.append({"a": a, "b": b, "c": c, "d": d})

        return StackedOrthogonalPolynomialKAN.init_params(key, dims, degrees, recurrences)

    @staticmethod
    def network_fn(all_params, x: jnp.ndarray) -> jnp.ndarray:
        return StackedOrthogonalPolynomialKAN.network_fn(all_params, x)


class StackedHermiteKAN_(Network):
    """
    Wrapper for a stacked Hermite‐polynomial KAN (physicists' version).
    Implemented concisely via vectorized array expressions.
    
    Args:
      dims:    [I₀, I₁, ..., I_B], length = B+1
      degrees: [D₀, D₁, ..., D_{B-1}], length = B
    """

    @staticmethod
    def init_params(
        key ,
        dims: List[int],
        degrees: List[int]
    ) -> Tuple[Dict, Dict]:
        B = len(degrees)
        assert len(dims) == B + 1

        # Split the PRNG key into B sub‐keys
        keys = jax.random.split(key, B)

        recurrences: List[Dict[str, jnp.ndarray]] = []
        for i, D in enumerate(degrees):
            n = jnp.arange(D + 1, dtype=jnp.float32)
            a = jnp.where(n >= 1.0, 2.0, 0.0)
            b = jnp.zeros(D + 1, dtype=jnp.float32)
            c = jnp.where(n >= 2.0, 2.0 * (n - 1.0), 0.0)
            d = jnp.ones(D + 1, dtype=jnp.float32)
            recurrences.append({"a": a, "b": b, "c": c, "d": d})

        return StackedOrthogonalPolynomialKAN.init_params(key, dims, degrees, recurrences)

    @staticmethod
    def network_fn(all_params, x: jnp.ndarray) -> jnp.ndarray:
        return StackedOrthogonalPolynomialKAN.network_fn(all_params, x)

class StackedJacobiKAN_(Network):
    """
    Wrapper for a stacked Jacobi‐polynomial KAN with α=β=1 (i.e. Jacobi(1,1)),
    implemented succinctly (no trainable α,β). Delegates to StackedOrthogonalPolynomialKAN.
    
    Recurrence (α=β=1):
        P₀(x) = 1,
        P₁(x) = 2x,
        Pₙ(x) = [ (2n+1)(2n+2)·x·P_{n-1}(x)  –  2n²(2n+2)·P_{n-2}(x) ]
               / [ 2n(n+2) ],  n ≥ 2.
    """

    @staticmethod
    def init_params(
        key,
        dims: List[int],
        degrees: List[int]
    ) -> Tuple[Dict, Dict]:
        B = len(degrees)
        assert len(dims) == B + 1

        # Split key into B subkeys (used for coeff init)
        keys = jax.random.split(key, B)

        recurrences: List[Dict[str, jnp.ndarray]] = []
        for i, D in enumerate(degrees):
            n = jnp.arange(D + 1, dtype=jnp.float32)

            # a[n] = 0 if n=0; 2 if n=1; ½·(2n+1)(2n+2) if n≥2
            a = jnp.where(
                n == 0.0,
                0.0,
                jnp.where(
                    n == 1.0,
                    2.0,
                    0.5 * (2.0 * n + 1.0) * (2.0 * n + 2.0)
                )
            )

            # b[n] = 0 (because α−β=0 when α=β=1)
            b = jnp.zeros(D + 1, dtype=jnp.float32)

            # c[n] = 0 if n<2; n^2 if n>=2  (since (1/2)(n)(n)(2n+2) = n²(n+1) but with α=β=1 it simplifies to n²)
            c = jnp.where(n < 2.0, 0.0, n * n)

            # d[n] = 1 if n<2; 2n(n+2) if n≥2
            d = jnp.where(
                n < 2.0,
                1.0,
                2.0 * n * (n + 2.0)
            )

            recurrences.append({"a": a, "b": b, "c": c, "d": d})

        return StackedOrthogonalPolynomialKAN.init_params(key, dims, degrees, recurrences)

    @staticmethod
    def network_fn(all_params, x: jnp.ndarray) -> jnp.ndarray:
        return StackedOrthogonalPolynomialKAN.network_fn(all_params, x)
