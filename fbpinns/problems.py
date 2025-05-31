"""
Defines PDE problems to solve

Each problem class must inherit from the Problem base class.
Each problem class must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import jax.nn
import jax.numpy as jnp
import numpy as np

from fbpinns.util.logger import logger
from fbpinns.traditional_solutions.analytical.burgers_solution import burgers_viscous_time_exact1
from fbpinns.traditional_solutions.seismic_cpml.seismic_CPML_2D_pressure_second_order import seismicCPML2D


class Problem:
    """Base problem class to be inherited by different problem classes.

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

        # below parameters need to be defined
        static_params = {
            "dims":None,# (ud, xd)# dimensionality of u and x
            }
        raise NotImplementedError

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        """Samples all constraints.
        Returns [[x_batch, *any_constraining_values, required_ujs], ...]. Each list element contains
        the x_batch points and any constraining values passed to the loss function, and the required
        solution and gradient components required in the loss function, for each constraint."""
        raise NotImplementedError

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        """Applies optional constraining operator"""
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        """Computes the PINN loss function, using constraints with the same structure output by sample_constraints"""
        raise NotImplementedError

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        """Defines exact solution, if it exists"""
        raise NotImplementedError

    @staticmethod
    def do_print(all_params, mse, attn_loss, phys):
        selected = all_params["trainable"]["problem"]["selected"].astype(jnp.int32)
        attention = all_params["trainable"]["attention"]["alpha"][selected]  # (N,1)
        current_i = all_params["trainable"]["problem"]["current_i"]

        jax.debug.print("curr_i = {i}, mse = {m1:.6f}, attn = {m2:.6f}", i=current_i, m1=mse, m2=attn_loss)
        jax.debug.print("residual max = {a}, attention head = {b}", a=jnp.max(jnp.abs(phys)), b=attention[:5, 0])
        return None

    @staticmethod
    def attention_print(all_params, mse, attn_loss, phys):
        # Conditionally invoke the debug-prints every 1000 steps
        current_i = all_params["trainable"]["problem"]["current_i"]
        _ = jax.lax.cond(current_i[0] % 1000 == 0, lambda _: Problem.do_print(all_params, mse, attn_loss, phys), lambda _: None, operand=None)




class HarmonicOscillator1D(Problem):
    """Solves the time-dependent damped harmonic oscillator
          d^2 u      du
        m ----- + mu -- + ku = 0
          dt^2       dt

        Boundary conditions:
        u (0) = 1
        u'(0) = 0
    """

    @staticmethod
    def init_params(d=2, w0=20):

        mu, k = 2*d, w0**2

        static_params = {
            "dims":(1,1),
            "d":d,
            "w0":w0,
            "mu":mu,
            "k":k,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(0,0))
        )

        # boundary loss
        x_batch_boundary = jnp.array([0.]).reshape((1,1))
        u_boundary = jnp.array([1.]).reshape((1,1))
        ut_boundary = jnp.array([0.]).reshape((1,1))
        required_ujs_boundary = (
            (0,()),
            (0,(0,)),
        )

        return [[x_batch_phys, required_ujs_phys], [x_batch_boundary, u_boundary, ut_boundary, required_ujs_boundary]]

    @staticmethod
    def loss_fn(all_params, constraints):

        mu, k = all_params["static"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        # physics loss
        _, u, ut, utt = constraints[0]
        phys = jnp.mean((utt + mu*ut + k*u)**2)

        # boundary loss
        _, uc, utc, u, ut = constraints[1]
        if len(uc):
            boundary = 1e6*jnp.mean((u-uc)**2) + 1e2*jnp.mean((ut-utc)**2)
        else:
            boundary = 0# if no boundary points are inside the active subdomains (i.e. u.shape[0]=0), jnp.mean returns nan

        return phys + boundary

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):

        d, w0 = all_params["static"]["problem"]["d"], all_params["static"]["problem"]["w0"]

        w = jnp.sqrt(w0**2-d**2)
        phi = jnp.arctan(-d/w)
        A = 1/(2*jnp.cos(phi))
        cos = jnp.cos(phi + w * x_batch)
        exp = jnp.exp(-d * x_batch)
        u = exp * 2 * A * cos

        return u


class HarmonicOscillator1DHardBC(HarmonicOscillator1D):
    """Solves the time-dependent damped harmonic oscillator using hard boundary conditions
          d^2 u      du
        m ----- + mu -- + ku = 0
          dt^2       dt

        Boundary conditions:
        u (0) = 1
        u'(0) = 0
    """

    @staticmethod
    def init_params(d=2, w0=20, sd=0.1):

        mu, k = 2*d, w0**2

        static_params = {
            "dims":(1,1),
            "d":d,
            "w0":w0,
            "mu":mu,
            "k":k,
            "sd":sd,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(0,0))
        )
        return [[x_batch_phys, required_ujs_phys],]# only physics loss required in this case

    @staticmethod
    def constraining_fn(all_params, x_batch, u):

        sd = all_params["static"]["problem"]["sd"]
        x, tanh = x_batch[:,0:1], jnp.tanh

        u = 1 + (tanh(x/sd)**2) * u# applies hard BCs
        return u

    @staticmethod
    def loss_fn(all_params, constraints):

        mu, k = all_params["static"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        # physics loss
        _, u, ut, utt = constraints[0]
        phys = jnp.mean((utt + mu*ut + k*u)**2)

        return phys


class HarmonicOscillator1DInverse(HarmonicOscillator1D):
    """Solves the time-dependent damped harmonic oscillator inverse problem
          d^2 u      du
        m ----- + mu -- + ku = 0
          dt^2       dt

        Boundary conditions:
        u (0) = 1
        u'(0) = 0
    """

    @staticmethod
    def init_params(d=2, w0=20):

        mu, k = 2*d, w0**2

        static_params = {
            "dims":(1,1),
            "d":d,
            "w0":w0,
            "mu_true":mu,
            "k":k,
            }
        trainable_params = {
            "mu":jnp.array(0.),# learn mu from constraints
            }

        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(0,0))
        )

        # data loss
        x_batch_data = jnp.linspace(0,1,13).astype(float).reshape((13,1))# use 13 observational data points
        u_data = HarmonicOscillator1DInverse.exact_solution(all_params, x_batch_data)
        required_ujs_data = (
            (0,()),
            )

        return [[x_batch_phys, required_ujs_phys], [x_batch_data, u_data, required_ujs_data]]

    @staticmethod
    def loss_fn(all_params, constraints):

        mu, k = all_params["trainable"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        # physics loss
        _, u, ut, utt = constraints[0]
        phys = jnp.mean((utt + mu*ut + k*u)**2)

        # data loss
        _, uc, u = constraints[1]
        data = 1e6*jnp.mean((u-uc)**2)

        return phys + data




class BurgersEquation2D(Problem):
    """Solves the time-dependent 1D viscous Burgers equation
        du       du        d^2 u
        -- + u * -- = nu * -----
        dt       dx        dx^2

        for -1.0 < x < +1.0, and 0 < t

        Boundary conditions:
        u(x,0) = - sin(pi*x)
        u(-1,t) = u(+1,t) = 0
    """

    @staticmethod
    def init_params(nu=0.01/jnp.pi, sd=0.1):

        static_params = {
            "dims":(1,2),
            "nu":nu,
            "sd":sd,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(1,)),
            (0,(0,0)),
        )
        return [[x_batch_phys, required_ujs_phys],]


    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sd = all_params["static"]["problem"]["sd"]
        x, t, tanh, sin, pi = x_batch[:,0:1], x_batch[:,1:2], jax.nn.tanh, jnp.sin, jnp.pi
        u = tanh((x+1)/sd)*tanh((1-x)/sd)*tanh((t-0)/sd)*u - sin(pi*x)
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        nu = all_params["static"]["problem"]["nu"]
        _, u, ux, ut, uxx = constraints[0]
        phys = ut + (u*ux) - (nu*uxx)
        return jnp.mean(phys**2), phys

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):

        nu = all_params["static"]["problem"]["nu"]

        # use the burgers_solution code to compute analytical solution
        xmin,xmax = x_batch[:,0].min().item(), x_batch[:,0].max().item()
        tmin,tmax = x_batch[:,1].min().item(), x_batch[:,1].max().item()
        vx = np.linspace(xmin,xmax,batch_shape[0])
        vt = np.linspace(tmin,tmax,batch_shape[1])
        logger.info("Running burgers_viscous_time_exact1..")
        vu = burgers_viscous_time_exact1(nu, len(vx), vx, len(vt), vt)
        u = jnp.array(vu.flatten()).reshape((-1,1))
        return u




class WaveEquationConstantVelocity3D(Problem):
    """Solves the time-dependent (2+1)D wave equation with constant velocity
        d^2 u   d^2 u    1  d^2 u
        ----- + ----- - --- ----- = 0
        dx^2    dy^2    c^2 dt^2

        Boundary conditions:
        u(x,y,0) = amp * exp( -0.5 (||[x,y]-mu||/sd)^2 )
        du
        --(x,y,0) = 0
        dt
    """

    @staticmethod
    def init_params(c0=1, source=np.array([[0., 0., 0.2, 1.]])):

        static_params = {
            "dims":(1,3),
            "c0":c0,
            "c_fn":WaveEquationConstantVelocity3D.c_fn,# velocity function
            "source":jnp.array(source),# location, width and amplitude of initial gaussian sources (k, 4)
            }
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
            (0,(1,1)),
            (0,(2,2)),
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        params = all_params["static"]["problem"]
        c0, source = params["c0"], params["source"]
        x, t = x_batch[:,0:2], x_batch[:,2:3]
        tanh, exp = jax.nn.tanh, jnp.exp

        # get starting wavefield
        p = jnp.expand_dims(source, axis=1)# (k, 1, 4)
        x = jnp.expand_dims(x, axis=0)# (1, n, 2)
        f = (p[:,:,3:4]*exp(-0.5 * ((x-p[:,:,0:2])**2).sum(2, keepdims=True)/(p[:,:,2:3]**2))).sum(0)# (n, 1)

        # form time-decaying anzatz
        t1 = source[:,2].min()/c0
        f = exp(-0.5*(1.5*t/t1)**2) * f
        t = tanh(2.5*t/t1)**2
        return f + t*u

    @staticmethod
    def loss_fn(all_params, constraints):
        c_fn = all_params["static"]["problem"]["c_fn"]
        x_batch, uxx, uyy, utt = constraints[0]
        phys = (uxx + uyy) - (1/c_fn(all_params, x_batch)**2)*utt
        return jnp.mean(phys**2), phys

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):

        # use the seismicCPML2D FD code with very fine sampling to compute solution

        params = all_params["static"]["problem"]
        c0, source = params["c0"], params["source"]
        c_fn = params["c_fn"]

        (xmin, ymin, tmin), (xmax, ymax, tmax) = np.array(x_batch.min(0)), np.array(x_batch.max(0))

        # get grid spacing
        deltax, deltay, deltat = (xmax-xmin)/(batch_shape[0]-1), (ymax-ymin)/(batch_shape[1]-1), (tmax-tmin)/(batch_shape[2]-1)

        # get f0, target deltas of FD simulation
        f0 = c0/source[:,2].min()# approximate frequency of wave
        DELTAX = DELTAY = 1/(f0*10)# target fine sampled deltas
        DELTAT = DELTAX / (4*np.sqrt(2)*c0)# target fine sampled deltas
        dx, dy, dt = int(np.ceil(deltax/DELTAX)), int(np.ceil(deltay/DELTAY)), int(np.ceil(deltat/DELTAT))# make sure deltas are a multiple of test deltas
        DELTAX, DELTAY, DELTAT = deltax/dx, deltay/dy, deltat/dt
        NX, NY, NSTEPS = batch_shape[0]*dx-(dx-1), batch_shape[1]*dy-(dy-1), batch_shape[2]*dt-(dt-1)

        # get starting wavefield
        xx,yy = np.meshgrid(np.linspace(xmin, xmax, NX), np.linspace(ymin, ymax, NY), indexing="ij")# (NX, NY)
        x = np.stack([xx.ravel(), yy.ravel()], axis=1)# (n, 2)
        exp = np.exp
        p = np.expand_dims(source, axis=1)# (k, 1, 4)
        x = np.expand_dims(x, axis=0)# (1, n, 2)
        f = (p[:,:,3:4]*exp(-0.5 * ((x-p[:,:,0:2])**2).sum(2, keepdims=True)/(p[:,:,2:3]**2))).sum(0)# (n, 1)
        p0 = f.reshape((NX, NY))

        # get velocity model
        x = np.stack([xx.ravel(), yy.ravel()], axis=1)# (n, 2)
        c = np.array(c_fn(all_params, x))
        if c.shape[0]>1: c = c.reshape((NX, NY))
        else: c = c*np.ones_like(xx)

        # add padded CPML boundary
        NPOINTS_PML = 10
        p0 = np.pad(p0, [(NPOINTS_PML,NPOINTS_PML),(NPOINTS_PML,NPOINTS_PML)], mode="edge")
        c =   np.pad(c, [(NPOINTS_PML,NPOINTS_PML),(NPOINTS_PML,NPOINTS_PML)], mode="edge")

        # run simulation
        logger.info(f'Running seismicCPML2D {(NX, NY, NSTEPS)}..')
        wavefields, _ = seismicCPML2D(
                    NX+2*NPOINTS_PML,
                    NY+2*NPOINTS_PML,
                    NSTEPS,
                    DELTAX,
                    DELTAY,
                    DELTAT,
                    NPOINTS_PML,
                    c,
                    np.ones((NX+2*NPOINTS_PML,NY+2*NPOINTS_PML)),
                    (p0.copy(),p0.copy()),
                    f0,
                    np.float32,
                    output_wavefields=True,
                    gather_is=None)

        # get croped, decimated, flattened wavefields
        wavefields = wavefields[:,NPOINTS_PML:-NPOINTS_PML,NPOINTS_PML:-NPOINTS_PML]
        wavefields = wavefields[::dt, ::dx, ::dy]
        wavefields = np.moveaxis(wavefields, 0, -1)
        assert wavefields.shape == batch_shape
        u = wavefields.reshape((-1, 1))

        return u

    @staticmethod
    def c_fn(all_params, x_batch):
        "Computes the velocity model"

        c0 = all_params["static"]["problem"]["c0"]
        return jnp.array([[c0]], dtype=float)# (1,1) scalar value


class WaveEquationGaussianVelocity3D(WaveEquationConstantVelocity3D):
    """Solves the time-dependent (2+1)D wave equation with gaussian mixture velocity
        d^2 u   d^2 u    1  d^2 u
        ----- + ----- - --- ----- = 0
        dx^2    dy^2    c^2 dt^2

        Boundary conditions:
        u(x,y,0) = amp * exp( -0.5 (||[x,y]-mu||/sd)^2 )
        du
        --(x,y,0) = 0
        dt
    """

    @staticmethod
    def init_params(c0=1, source=np.array([[0., 0., 0.2, 1.]]), mixture=np.array([[0.5, 0.5, 1., 0.2]])):

        static_params = {
            "dims":(1,3),
            "c0":c0,
            "c_fn":WaveEquationGaussianVelocity3D.c_fn,# velocity function
            "source":jnp.array(source),# location, width and amplitude of initial gaussian sources (k, 4)
            "mixture":jnp.array(mixture),# location, width and amplitude of gaussian pertubations in velocity model (l, 4)
            }
        return static_params, {}

    @staticmethod
    def c_fn(all_params, x_batch):
        "Computes the velocity model"

        c0, mixture = all_params["static"]["problem"]["c0"], all_params["static"]["problem"]["mixture"]
        x = x_batch[:,0:2]# (n, 2)
        exp = jnp.exp

        # get velocity model
        p = jnp.expand_dims(mixture, axis=1)# (l, 1, 4)
        x = jnp.expand_dims(x, axis=0)# (1, n, 2)
        f = (p[:,:,3:4]*exp(-0.5 * ((x-p[:,:,0:2])**2).sum(2, keepdims=True)/(p[:,:,2:3]**2))).sum(0)# (n, 1)
        c = c0 + f# (n, 1)
        return c
    
class HarmonicOscillator1D(Problem):
    """Solves the time-dependent damped harmonic oscillator using hard boundary conditions
          d^2 u      du
        m ----- + mu -- + ku = 0
          dt^2       dt

        Boundary conditions:
        u (0) = 1
        u'(0) = 0
    """

    @staticmethod
    def init_params(d=2, w0=20, sd=0.1):

        mu, k = 2*d, w0**2

        static_params = {
            "dims":(1,1),
            "d":d,
            "w0":w0,
            "mu":mu,
            "k":k,
            "sd":sd,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(0,0))
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):

        sd = all_params["static"]["problem"]["sd"]
        x, tanh = x_batch[:,0:1], jnp.tanh

        u = 1 + (tanh(x/sd)**2) * u# applies hard BCs
        return u

    @staticmethod
    def loss_fn(all_params, constraints):

        mu, k = all_params["static"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        _, u, ut, utt = constraints[0]
        phys = utt + mu*ut + k*u
        mse = jnp.mean((phys)**2)

        return mse, phys

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):

        d, w0 = all_params["static"]["problem"]["d"], all_params["static"]["problem"]["w0"]

        w = jnp.sqrt(w0**2-d**2)
        phi = jnp.arctan(-d/w)
        A = 1/(2*jnp.cos(phi))
        cos = jnp.cos(phi + w * x_batch)
        exp = jnp.exp(-d * x_batch)
        u = exp * 2 * A * cos

        return u

class HarmonicOscillator1DAttention(HarmonicOscillator1D):

    @staticmethod
    def loss_fn(all_params, constraints):

        mu, k = all_params["static"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        _, u, ut, utt = constraints[0]
        phys = utt + mu*ut + k*u
        mse = jnp.mean((phys)**2)

        Problem.attention_print(all_params, mse, 0, phys)

        return mse, phys


class HarmonicOscillator1D_MultiFreq(Problem):
    """Solves the time-dependent damped harmonic oscillator using hard boundary conditions
          d^2 u      du
        m ----- + mu -- + ku = 4cos(w0*t) + 40cos(w1*t)
          dt^2       dt

        Boundary conditions:
        u (0) = 0
        u'(0) = 0
    """

    @staticmethod
    def init_params(m=0, mu=1, k=0, w0=40, w1=40, sd=0.1):

        static_params = {
            "dims":(1,1),
            "m":m,
            "mu":mu,
            "k":k,
            "sd":sd,
            "w0":w0,
            "w1":w1,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(0,0))
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):

        sd = all_params["static"]["problem"]["sd"]
        x, tanh = x_batch[:,0:1], jnp.tanh

        u = (tanh(x/sd)**2) * u
        return u

    @staticmethod
    def loss_fn(all_params, constraints):

        m, mu, k = all_params["static"]["problem"]["m"], all_params["static"]["problem"]["mu"], all_params["static"]["problem"]["k"]
        w0, w1 = all_params["static"]["problem"]["w0"], all_params["static"]["problem"]["w1"]
        x_batch, u, ut, utt = constraints[0]
        t = x_batch[:,0:1]

        phys = m*utt + mu*ut + k*u - (w0*jnp.cos(w0*t) + w1*jnp.cos(w1*t))
        loss = jnp.mean((phys)**2)
        return loss, phys
    
    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        x, sin = x_batch[:,0:1], jnp.sin
        w0, w1 = all_params["static"]["problem"]["w0"], all_params["static"]["problem"]["w1"]
        u = sin(w0*x) + sin(w1*x)
        return u


class HeatEquation1D(Problem):
    """
    Solves the 1D heat equation:
    
        u_t = α u_xx

    on the domain x ∈ [0, 1] and t ∈ [0, T],
    with homogeneous Dirichlet boundary conditions:
        u(0,t) = u(1,t) = 0
    and initial condition:
        u(x,0) = sin(πx)

    The analytical solution is:
        u(x,t) = sin(πx) * exp(-α π² t)
    """

    @staticmethod
    def init_params(alpha=1.0, N=40000):
        static_params = {
            "dims": (1, 2), 
            "alpha": alpha,
        }
        trainable_params = {
            'attention': jnp.zeros((N, 1))
            }
        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        
        required_ujs_phys = (
            (0, (0, 0)),  # u_xx
            (0, (1,)),    # u_t
        )
        
        return [[x_batch_phys, required_ujs_phys]]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        """
        Enforces the Dirichlet BCs and the initial condition.
        Assumes x_batch has two columns: x and t.
        Constructs the solution as:
            u(x,t) = x*(1-x)*t * u + sin(πx)
        so that:
            u(x,0) = sin(πx),
            u(0,t) = 0, and
            u(1,t) = 0.
        """
        x = x_batch[:, 0:1]
        t = x_batch[:, 1:2]
        tanh = jax.nn.tanh
        sd = 0.1
        return tanh((-x)/sd)*tanh((1-x)/sd)*tanh((t)/sd) * u + jnp.sin(jnp.pi * x)

    @staticmethod
    def loss_fn(all_params, constraints):
        _, uxx, ut = constraints[0]
        alpha = all_params["static"]["problem"]["alpha"]
        
        residual = ut - alpha * uxx

        return jnp.mean(residual ** 2), residual

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        alpha = all_params["static"]["problem"]["alpha"]
        x = x_batch[:, 0:1]
        t = x_batch[:, 1:2]
        return jnp.sin(jnp.pi * x) * jnp.exp(-alpha * (jnp.pi ** 2) * t)


class Poisson2D(Problem):
    """
    Solves the 2D Poisson equation
        - u_xx - u_yy = f(x,y)
    on the domain [0,1] with Dirichlet boundary conditions u = 0 on ∂Ω.

    We choose f(x,y) such that the exact solution is:
        u(x,y) = sin(πx)sin(πy)
    which implies f(x,y) = 2π²sin(πx)sin(πy).
    """

    @staticmethod
    def init_params(f_coeff=2 * jnp.pi ** 2, sd=0.1, N=10000):
        # 'dims': (ud, xd) => u is scalar (ud=1) and x is 2D (xd=2)
        static_params = {
            "dims": (1, 2),
            "f_coeff": f_coeff,  # coefficient in the forcing function f(x,y)
            "sd": sd,
            'statictest': 1
        }
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # --- Physics loss: sample interior points ---
        # x_batch_phys: an array of shape (n_phys, 2)
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0, (0, 0)), # u_xx
            (0, (1, 1)), # u_yy
        )

        return [[x_batch_phys, required_ujs_phys],]
    
    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sd = all_params["static"]["problem"]["sd"]
        x, y, tanh = x_batch[:,0:1], x_batch[:,1:2], jax.nn.tanh
        u = tanh((x)/sd) * tanh((1-x)/sd) * tanh((y)/sd) * tanh((1-y)/sd) * u
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        # --- Physics loss ---
        # For the physics group, the constraints have been replaced with the evaluated quantities:
        x_phys, u_xx, u_yy = constraints[0]
        x, y = x_phys[:, 0:1], x_phys[:,1:2]

        f_coeff = all_params["static"]["problem"]["f_coeff"]
        f_val = f_coeff * jnp.sin(jnp.pi* x) * jnp.sin(jnp.pi* y)

        phys_residual = u_xx + u_yy + f_val

        return jnp.mean(phys_residual**2), phys_residual

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        # The exact solution is u(x,y) = sin(πx) sin(πy)
        x, y = x_batch[:, 0:1], x_batch[:,1:2]
        u = jnp.sin(jnp.pi* x) * jnp.sin(jnp.pi* y)
        return u


class Schrodinger1D_Stationary(Problem):
    
    @staticmethod
    def init_params(omega=1.0, L=5.0, sd=0.1):
        static_params = {
            "dims": (2, 2),
            "omega": omega,
            "L": L,
            "sd": sd
        }
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        x_batch = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        
        required_diffs = (
            (0, ()),     # u
            (0, (0, 0)), # u_xx
            (1, (1,)),   # v_t
            (1, ()),     # v
            (1, (0, 0)), # v_xx
            (0, (1,)),   # u_t
        )
        return [[x_batch, required_diffs],]

    @staticmethod
    def constraining_fn(all_params, x_batch, net_out):
        x = x_batch[:, 0:1]
        t = x_batch[:, 1:2]
        omega = all_params["static"]["problem"]["omega"]
        L = all_params["static"]["problem"]["L"]
        sd = all_params["static"]["problem"]["sd"]
        tanh = jax.nn.tanh
        
        # Constraining function
        c = tanh((L+x)/sd) * tanh((L-x)/sd)
        
        # Define the initial condition (ground state of the harmonic oscillator)
        psi0 = (omega / jnp.pi)**0.25 * jnp.exp(-omega * x**2 / 2.0)
        
        # Split the raw network output into its two components.
        u = net_out[:, 0:1]
        v = net_out[:, 1:2]
        
        # Reparameterize to enforce the conditions.
        u = c * t * u + psi0
        v = c * t * v
        # Concatenate u and v to form the full two-component output.
        return jnp.concatenate([u, v], axis=1)

    @staticmethod
    def loss_fn(all_params, constraints):
        """
        Computes the physics residual for both equations:
        
        Equation (1): -v_t + ½ u_{xx} - ½ ω² x² u = 0.
        Equation (2):  u_t + ½ v_{xx} - ½ ω² x² v = 0.
        
        The loss is defined as the sum of the mean squared errors of the residuals for
        both equations.
        """
        omega = all_params["static"]["problem"]["omega"]
        x_batch, u, uxx, vt, v, vxx, ut = constraints[0]
        
        x = x_batch[:, 0:1]
        
        res1 = -vt + 0.5 * uxx - 0.5 * omega**2 * (x**2) * u
        res2 = ut + 0.5 * vxx - 0.5 * omega**2 * (x**2) * v
        
        loss1 = jnp.mean(res1**2)
        loss2 = jnp.mean(res2**2)
        return loss1 + loss2, res1 + res2

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        """
        Returns the exact ground-state solution for the harmonic oscillator.
        
        The analytical solution (up to a global phase) is given by:
            u(x,t) = ψ₀(x) cos(ω t/2),
            v(x,t) = -ψ₀(x) sin(ω t/2),
        where ψ₀(x) = (ω/π)^(¼) exp(-ω x²/2).
        """
        omega = all_params["static"]["problem"]["omega"]
        x = x_batch[:, 0:1]
        t = x_batch[:, 1:2]
        psi0 = (omega / jnp.pi)**0.25 * jnp.exp(-omega * x**2 / 2.0)
        u = psi0 * jnp.cos(omega * t / 2.0)
        v = -psi0 * jnp.sin(omega * t / 2.0)
        return jnp.concatenate([u, v], axis=1)


class Schrodinger1D_Non_Stationary(Schrodinger1D_Stationary):
    
    # @staticmethod
    # def init_params(omega=1.0, L=5.0, sd=0.1):
    #     static_params = {
    #         "dims": (2, 2),
    #         "omega": omega,
    #         "L": L,
    #         "sd": sd
    #     }
    #     return static_params, {}

    # @staticmethod
    # def sample_constraints(all_params, domain, key, sampler, batch_shapes):
    #     x_batch = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        
    #     required_diffs = (
    #         (0, ()),     # u
    #         (0, (0, 0)), # u_xx
    #         (1, (1,)),   # v_t
    #         (1, ()),     # v
    #         (1, (0, 0)), # v_xx
    #         (0, (1,)),   # u_t
    #     )
    #     return [[x_batch, required_diffs],]

    @staticmethod
    def constraining_fn(all_params, x_batch, net_out):
        """
        Enforces both the initial condition and the boundary conditions.
        """

        x = x_batch[:, 0:1]
        t = x_batch[:, 1:2]
        omega = all_params["static"]["problem"]["omega"]
        L = all_params["static"]["problem"]["L"]
        sd = all_params["static"]["problem"]["sd"]
        tanh = jax.nn.tanh
        
        # Constraining function
        c = tanh((L+x)/sd) * tanh((L-x)/sd)
        
        # Define the initial condition (ground state of the harmonic oscillator)
        psi0 = 2.0/jnp.cosh(x)
        
        # Split the raw network output into its two components.
        u = net_out[:, 0:1]
        v = net_out[:, 1:2]
        
        # Reparameterize to enforce the conditions.
        u = c * t * u + psi0
        v = c * t * v
        # Concatenate u and v to form the full two-component output.
        return jnp.concatenate([u, v], axis=1)

    @staticmethod
    def loss_fn(all_params, constraints):
        omega = all_params["static"]["problem"]["omega"]
        x_batch, u, uxx, vt, v, vxx, ut = constraints[0]
        
        x = x_batch[:, 0:1]
        
        res1 = -vt + 0.5 * uxx + (u**2 + v**2) * u
        res2 = ut + 0.5 * vxx + (u**2 + v**2) * v
        
        loss1 = jnp.mean(res1**2)
        loss2 = jnp.mean(res2**2)
        return loss1 + loss2, res1+res2

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        """
        Need to find exact solution. Returns zeros.
        """
        omega = all_params["static"]["problem"]["omega"]
        x = x_batch[:, 0:1]
        t = x_batch[:, 1:2]
        sech = lambda z: 1.0/jnp.cosh(z)
        u = omega * sech(omega * x) * jnp.cos( (omega**2 * t)/2.0 )
        v = omega * sech(omega * x) * jnp.sin( (omega**2 * t)/2.0 )
        return jnp.zeros_like(jnp.concatenate([u, v], axis=1))
        

class WaveEquation2D(Problem):
    """Solves the time-dependent 1D viscous Burgers equation
        d^2 u       d^2 u    
        ----- - c^2 ----- = 0
        d t^2       d x^2    

        for (x, t) in [0, 1]^2

        Boundary conditions:
        u(0, t) = 0
        u(1, t) = 0
        u(x, 0) = sin(πx) + 0.5 sin(4πx)
        u_t(x, 0) = 0
    """

    @staticmethod
    def init_params(c=jnp.sqrt(2), sd=0.1):

        static_params = {
            "dims":(1,2),
            "c":c,
            "sd":sd,
            }
        
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
            (0,(1,1)),
        )
        return [[x_batch_phys, required_ujs_phys],]


    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sd = all_params["static"]["problem"]["sd"]
        x, t, tanh, sin, pi = x_batch[:,0:1], x_batch[:,1:2], jax.nn.tanh, jnp.sin, jnp.pi
        u = tanh(x/sd) * tanh((1-x)/sd) * t**2 * u  + (sin(pi*x) + 0.5 * sin(4*pi*x))
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        c = all_params["static"]["problem"]["c"]
        _, uxx, utt = constraints[0]
        phys = utt - c**2 * uxx
        mse = jnp.mean(phys**2)
        return mse, phys
    
    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):

        c = all_params["static"]["problem"]["c"]
        x, t, sin, cos, pi = x_batch[:,0:1], x_batch[:,1:2], jnp.sin, jnp.cos, jnp.pi
        u = sin(pi*x)*cos(c*pi*t) + 0.5 * sin(4*pi*x)*cos(4*c*pi*t)
        return u


class KovasznayFlow(Problem):
    """
    Solves the steady 2D incompressible Navier–Stokes equations (momentum + continuity)
    via the Kovasznay flow, defined on the domain [0,1] x [0,1].

    The equations are:
        u u_x + v u_y + p_x - ν (u_{xx}+u_{yy}) = 0,
        u v_x + v v_y + p_y - ν (v_{xx}+v_{yy}) = 0,
        u_x + v_y = 0,
    where ν is the kinematic viscosity.

    The exact solution is given by:
        u(x,y) = 1 - e^(λ x) cos(2π y),
        v(x,y) = (λ/(2π)) e^(λ x) sin(2π y),
        p(x,y) = ½ (1 - e^(2λ x)),
    with
        λ = Re/2 - sqrt((Re/2)^2+4π²),
        Re = 1/ν.
        
    For example, for Re=40 (ν=0.025) we have λ ≈ -0.952.
    
    We enforce the boundary conditions by reparameterizing the solution:
      u(x,y) = M(x,y)*N_u(x,y) + u_exact(x,y),
      v(x,y) = M(x,y)*N_v(x,y) + v_exact(x,y),
      p(x,y) = N_p(x,y) + p_exact(x,y),
    where M(x,y)=x(1-x)y(1-y) vanishes on ∂([0,1]×[0,1]).
    """

    @staticmethod
    def init_params(nu=0.025, sd=0.1):
        # Compute Reynolds number and λ
        Re = 1.0 / nu
        lam = Re / 2.0 - jnp.sqrt((Re/2.0)**2 + 4 * (jnp.pi**2))
        static_params = {
            "dims": (3, 2),  # 3 outputs: u, v, p; 2 inputs: x, y.
            "nu": nu,
            "lam": lam,
            "sd": sd,
        }
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # Sample interior points from the domain [0,1] x [0,1]
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        # Request second derivatives for u and v, and first derivatives for p,
        # as well as first derivatives for the divergence condition.
        # Here we request:
        #   (0, (0,)) for u_x and (0, (1,)) for u_y,
        #   (0, (0,0)) for u_xx and (0, (1,1)) for u_yy,
        #   similarly for v,
        #   (2, (0,)) for p_x and (2, (1,)) for p_y.
        # To also enforce continuity u_x+v_y=0, we use the already requested u_x and v_y.
        required_diffs = (
            (0, ()),       # u
            (0, (0,)),     # u_x
            (0, (1,)),     # u_y
            (0, (0,0)),    # u_xx
            (0, (1,1)),    # u_yy
            (1, ()),       # v
            (1, (0,)),     # v_x
            (1, (1,)),     # v_y
            (1, (0,0)),    # v_xx
            (1, (1,1)),    # v_yy
            (2, ()),       # p
            (2, (0,)),     # p_x
            (2, (1,)),     # p_y
        )
        return [[x_batch_phys, required_diffs]]

    @staticmethod
    def constraining_fn(all_params, x_batch, net_out):
        """
        Reparameterizes the network output to enforce the boundary conditions.
        
        Let the raw network output be:
            N(x,y) = [N_u(x,y), N_v(x,y), N_p(x,y)].
        Then define the full solution as:
            u(x,y) = M(x,y)*N_u(x,y) + u_exact(x,y),
            v(x,y) = M(x,y)*N_v(x,y) + v_exact(x,y),
            p(x,y) = N_p(x,y) + p_exact(x,y),
        with the multiplier M(x,y)=x(1-x)y(1-y) which vanishes on the boundary.
        """
        x = x_batch[:, 0:1]
        y = x_batch[:, 1:2]
        sd, tanh = all_params["static"]["problem"]["sd"], jax.nn.tanh
        # M = x * (1 - x) * y * (1 - y)
        c = tanh((x)/sd) * tanh((1-x)/sd) * tanh((y)/sd) * tanh((1-y)/sd)
        
        # Extract exact solution (using lam and nu from static_params)
        lam = all_params["static"]["problem"]["lam"]
        pi = jnp.pi
        u_exact = 1 - jnp.exp(lam * x) * jnp.cos(2 * pi * y)
        v_exact = (lam / (2 * pi)) * jnp.exp(lam * x) * jnp.sin(2 * pi * y)
        p_exact = 0.5 * (1 - jnp.exp(2 * lam * x))
        
        # Split network outputs
        N_u = net_out[:, 0:1]
        N_v = net_out[:, 1:2]
        N_p = net_out[:, 2:3]
        
        u = c * N_u + u_exact
        v = c * N_v + v_exact
        p = N_p + p_exact
        return jnp.concatenate([u, v, p], axis=1)

    @staticmethod
    def loss_fn(all_params, constraints):
        # Retrieve the kinematic viscosity from the parameters.
        nu = all_params["static"]["problem"]["nu"]
        
        # Unpack the constraints array.
        # We assume constraints[0] contains:
        # [x_batch, u, u_x, u_y, u_xx, u_yy, v, v_x, v_y, v_xx, v_yy, p, p_x, p_y]
        (x_batch, u, u_x, u_y, u_xx, u_yy,
        v, v_x, v_y, v_xx, v_yy,
        p, p_x, p_y) = constraints[0]
        
        # Compute the momentum residuals.
        # Residual for the u-momentum equation:
        # R_u = u * u_x + v * u_y + p_x - nu*(u_xx + u_yy)
        residual_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        
        # Residual for the v-momentum equation:
        # R_v = u * v_x + v * v_y + p_y - nu*(v_xx + v_yy)
        residual_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        
        # Compute the continuity residual:
        # R_c = u_x + v_y
        residual_c = u_x + v_y
        
        # Define the total loss as the sum of mean squared errors.
        loss_u = jnp.mean(residual_u ** 2)
        loss_v = jnp.mean(residual_v ** 2)
        loss_c = jnp.mean(residual_c ** 2)
        
        return loss_u + loss_v + loss_c, residual_u + residual_v + residual_c


    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        """
        Returns the exact Kovasznay flow solution:
            u(x,y) = 1 - e^(λ x) cos(2π y),
            v(x,y) = (λ/(2π)) e^(λ x) sin(2π y),
            p(x,y) = ½ (1 - e^(2λ x)).
        """
        lam = all_params["static"]["problem"]["lam"]
        pi = jnp.pi
        x = x_batch[:, 0:1]
        y = x_batch[:, 1:2]
        u = 1 - jnp.exp(lam * x) * jnp.cos(2 * pi * y)
        v = (lam / (2 * pi)) * jnp.exp(lam * x) * jnp.sin(2 * pi * y)
        p = 0.5 * (1 - jnp.exp(2 * lam * x))
        return jnp.concatenate([u, v, p], axis=1)
    
class TaylorGreen3DFlow(Problem):
    """
    Steady 3D incompressible Navier–Stokes on Ω = [-1,1]^3 with a manufactured
    Taylor–Green vortex solution mapping (x,y,z) → (u,v,w,p) ∈ R^4.
    
    PDE system:
        (u·∇)u + ∇p − ν Δu = f,    ∇·u = 0,
    where u = (u,v,w), p is pressure, and ν is viscosity.

    Manufactured exact solution:
        u(x,y,z) =  sin(πx) cos(πy) cos(πz),
        v(x,y,z) = -cos(πx) sin(πy) cos(πz),
        w(x,y,z) =  0,
        p(x,y,z) = ¼ [cos(2πx) + cos(2πy)] cos(2πz).
    Forcing f is chosen so that (u,p) satisfy the PDE exactly.

    Boundary conditions are enforced via
        u = M·N_u + u_exact, etc.,
    with M(x,y,z)=tanh((x+1)/sd)·tanh((1−x)/sd)·tanh((y+1)/sd)·tanh((1−y)/sd)
                 ·tanh((z+1)/sd)·tanh((1−z)/sd), so M|∂Ω = 0.
    """

    @staticmethod
    def init_params(nu: float = 0.01, sd: float = 0.1):
        """
        ν: kinematic viscosity
        sd: smoothing parameter for the boundary multiplier
        """
        static_params = {
            "dims": (4, 3),  # 4 outputs (u,v,w,p), 3 inputs (x,y,z)
            "nu": nu,
            "sd": sd,
        }
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # sample interior collocation points in [-1,1]^3
        x_batch = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        # request derivatives for u,v,w,p
        # outputs: 0=u, 1=v, 2=w, 3=p
        required = []
        for i in range(3):  # for u,v,w
            required += [
                (i, ()),        # function itself
                (i, (0,)),      # ∂/∂x_i=0 for u_x, etc.
                (i, (1,)),
                (i, (2,)),
                (i, (0,0)),     # ∂²/∂x², etc.
                (i, (1,1)),
                (i, (2,2)),
            ]
        # pressure: only need p, p_x, p_y, p_z
        required += [
            (3, ()),
            (3, (0,)),
            (3, (1,)),
            (3, (2,)),
        ]
        return [[x_batch, tuple(required)]]

    @staticmethod
    def constraining_fn(all_params, x_batch, net_out):
        """
        Enforce exact boundary values via multiplier M.
        net_out: [N_u, N_v, N_w, N_p]
        """
        sd = all_params["static"]["problem"]["sd"]
        tanh = jax.nn.tanh
        x, y, z = x_batch[:, 0:1], x_batch[:, 1:2], x_batch[:, 2:3]

        # boundary multiplier M(x,y,z)
        M = (
            tanh((x + 1) / sd) * tanh((1 - x) / sd) *
            tanh((y + 1) / sd) * tanh((1 - y) / sd) *
            tanh((z + 1) / sd) * tanh((1 - z) / sd)
        )

        pi = jnp.pi
        # exact solution
        u_ex =  jnp.sin(pi * x) * jnp.cos(pi * y) * jnp.cos(pi * z)
        v_ex = -jnp.cos(pi * x) * jnp.sin(pi * y) * jnp.cos(pi * z)
        w_ex =  jnp.zeros_like(x)
        p_ex = 0.25 * (jnp.cos(2*pi*x) + jnp.cos(2*pi*y)) * jnp.cos(2*pi*z)

        N_u, N_v, N_w, N_p = (
            net_out[:, 0:1],
            net_out[:, 1:2],
            net_out[:, 2:3],
            net_out[:, 3:4],
        )

        u = M * N_u + u_ex
        v = M * N_v + v_ex
        w = M * N_w + w_ex
        p =    N_p + p_ex

        return jnp.concatenate([u, v, w, p], axis=1)

    @staticmethod
    def loss_fn(all_params, constraints):
        """
        Compute MSE of momentum and continuity residuals.
        """
        nu = all_params["static"]["problem"]["nu"]
        (x, 
         u, u_x, u_y, u_z, u_xx, u_yy, u_zz,
         v, v_x, v_y, v_z, v_xx, v_yy, v_zz,
         w, w_x, w_y, w_z, w_xx, w_yy, w_zz,
         p, p_x, p_y, p_z) = constraints[0]

        # momentum residuals
        r_u = u*u_x + v*u_y + w*u_z + p_x - nu*(u_xx + u_yy + u_zz)
        r_v = u*v_x + v*v_y + w*v_z + p_y - nu*(v_xx + v_yy + v_zz)
        r_w = u*w_x + v*w_y + w*w_z + p_z - nu*(w_xx + w_yy + w_zz)
        # continuity
        r_c = u_x + v_y + w_z

        # mean-squared
        loss = (
            jnp.mean(r_u**2) +
            jnp.mean(r_v**2) +
            jnp.mean(r_w**2) +
            jnp.mean(r_c**2)
        )

        # diagnostic residuals stacked per point
        resid = jnp.concatenate([r_u, r_v, r_w, r_c], axis=1)
        return loss, resid

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        """
        Return the manufactured Taylor–Green vortex.
        """
        pi = jnp.pi
        x, y, z = x_batch[:, 0:1], x_batch[:, 1:2], x_batch[:, 2:3]

        u =  jnp.sin(pi * x) * jnp.cos(pi * y) * jnp.cos(pi * z)
        v = -jnp.cos(pi * x) * jnp.sin(pi * y) * jnp.cos(pi * z)
        w =  jnp.zeros_like(x)
        p = 0.25 * (jnp.cos(2*pi*x) + jnp.cos(2*pi*y)) * jnp.cos(2*pi*z)

        return jnp.concatenate([u, v, w, p], axis=1)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from fbpinns.domains import RectangularDomainND

    np.random.seed(0)

    mixture=np.concatenate([
        np.random.uniform(-3, 3, (100,2)),# location
        0.4*np.ones((100,1)),# width
        0.3*np.random.uniform(-1, 1, (100,1)),# amplitude
        ], axis=1)

    source=np.array([# multiscale sources
        [0,0,0.1,1],
        [1,1,0.2,0.5],
        [-1.1,-0.5,0.4,0.25],
        ])

    # test wave equation
    for problem, kwargs in [(WaveEquationConstantVelocity3D, dict()),
                            (WaveEquationGaussianVelocity3D, dict(source=source, mixture=mixture))]:

        ps_ = problem.init_params(**kwargs)
        all_params = {"static":{"problem":ps_[0]}, "trainable":{"problem":ps_[1]}}

        batch_shape = (80,80,50)
        x_batch = RectangularDomainND._rectangle_samplerND(None, "grid", np.array([-3, -3, 0]), np.array([3, 3, 3]), batch_shape)

        plt.figure()
        c = np.array(problem.c_fn(all_params, x_batch))
        if c.shape[0]>1: c = c.reshape(batch_shape)
        else: c = c*np.ones(batch_shape)
        plt.imshow(c[:,:,0])
        plt.colorbar()
        plt.show()

        u = problem.exact_solution(all_params, x_batch, batch_shape).reshape(batch_shape)
        uc = np.zeros_like(x_batch)[:,0:1]
        uc = problem.constraining_fn(all_params, x_batch, uc).reshape(batch_shape)

        its = range(0,50,3)
        for u_ in [u, uc]:
            vmin, vmax = np.quantile(u, 0.05), np.quantile(u, 0.95)
            plt.figure(figsize=(2*len(its),5))
            for iplot,i in enumerate(its):
                plt.subplot(1,len(its),1+iplot)
                plt.imshow(u_[:,:,i], vmin=vmin, vmax=vmax)
            plt.show()
        plt.figure()
        plt.plot(u[40,40,:], label="u")
        plt.plot(uc[40,40,:], label="uc")
        t = np.linspace(0,1,50)
        plt.plot(np.tanh(2.5*t/(all_params["static"]["problem"]["source"][:,2].min()/all_params["static"]["problem"]["c0"]))**2)
        plt.legend()
        plt.show()




