from fbpinns.problems import (
    HarmonicOscillator1D, 
    HarmonicOscillator1D_MultiFreq, 
    HeatEquation1D, 
    BurgersEquation2D, 
    Poisson2D,
    Schrodinger1D_Stationary,
    Schrodinger1D_Non_Stationary,
    WaveEquation2D,
    KovasznayFlow,
    TaylorGreen3DFlow
    )
from operator import mul
from functools import reduce

attention_params_1 = lambda n: dict(
        eta_lr=1e-2,
        gamma_decay=0.99,
        out_dim=1,
        N=n
    )

# Harmonic Oscillator
# ------------------------------------------------------
HarmonicOscillator1D_N = 1000
HarmonicOscillator1D_Hyperparameters = dict(
    n_steps=50000,
    n_test=(HarmonicOscillator1D_N,),
    ns=((HarmonicOscillator1D_N,),),
    attention_tracking_kwargs=attention_params_1(HarmonicOscillator1D_N),
    optimiser_kwargs=dict(
        learning_rate=0.001
    ),
)

HarmonicOscillator1D_Config = dict(d=4, w0=80)
HarmonicOscillator1D_LowFreq = (HarmonicOscillator1D, 
                                HarmonicOscillator1D_Config,
                                HarmonicOscillator1D_Hyperparameters)

HarmonicOscillator1D_MF_Config = dict(m=0, mu=1, k=0, w0=4, w1=40, sd=0.1)
HarmonicOscillator1D_HighFreq = (HarmonicOscillator1D_MultiFreq, 
                                HarmonicOscillator1D_MF_Config,
                                HarmonicOscillator1D_Hyperparameters)


# Heat Equation
# ------------------------------------------------------
HeatEquation1D_N = (100, 100)
Heat_Eq_1_plus_1D_Config = dict(alpha=1.0, N=100*100)
Heat_Eq_1D_Hyperparameters = dict(
    n_steps=10000,
    ns=(HeatEquation1D_N,),
    n_test=HeatEquation1D_N,
    attention_tracking_kwargs=attention_params_1(reduce(mul, HeatEquation1D_N)),
    optimiser_kwargs=dict(
        learning_rate=0.001
    ),
)
Heat_Eq_1_plus_1D = (HeatEquation1D, 
               Heat_Eq_1_plus_1D_Config,
               Heat_Eq_1D_Hyperparameters)

# Burgers Equation
# ------------------------------------------------------
BurgersEquation1D_N = (200, 100)
Burgers_1_plus_1D_Config = dict()
Burgers_1_plus_1D_Hyperparameters = dict(
    n_steps=20000,
    ns=(BurgersEquation1D_N,),
    n_test=BurgersEquation1D_N,
    attention_tracking_kwargs=attention_params_1(reduce(mul, BurgersEquation1D_N)),
    optimiser_kwargs=dict(
        learning_rate=0.001
    ),
)
Burgers_1_plus_1D = (BurgersEquation2D, 
               Burgers_1_plus_1D_Config,
               Burgers_1_plus_1D_Hyperparameters)

# Poisson Equation
# ------------------------------------------------------
Poisson2D_N = (100, 100)
Poisson_2D_Config = dict()
Poisson_2D_Hyperparameters = dict(
    n_steps=10000,
    ns=(Poisson2D_N,),
    n_test=Poisson2D_N,
    attention_tracking_kwargs=attention_params_1(reduce(mul, Poisson2D_N)),
    optimiser_kwargs=dict(
        learning_rate=0.001
    ),
)
Poisson_2D = (Poisson2D, 
               Poisson_2D_Config,
               Poisson_2D_Hyperparameters)


# Schrodinger Equation 1D Time dependent
# ------------------------------------------------------
Schrodinger_N = (200,50)
Schrodinger_Config = dict()
Schrodinger_Hyperparameters = dict(
    n_steps=20000,
    ns=(Schrodinger_N,),
    n_test=Schrodinger_N,
    attention_tracking_kwargs=attention_params_1(reduce(mul, Schrodinger_N)),
    optimiser_kwargs=dict(
        learning_rate=0.001
    ),
)
Schrodinger1D_Stationary = (
    Schrodinger1D_Stationary,
    Schrodinger_Config,
    Schrodinger_Hyperparameters
)
Schrodinger1D_NonStationary = (
    Schrodinger1D_Non_Stationary,
    Schrodinger_Config,
    Schrodinger_Hyperparameters
)

# Wave eq 1+1D
# ------------------------------------------------------
Wave_N = (200,200)
Wave_Config = dict()
Wave_Hyperparameters = dict(
    n_steps=25000,
    ns=(Wave_N,),
    n_test=Wave_N,
    attention_tracking_kwargs=attention_params_1(reduce(mul, Wave_N)),
    optimiser_kwargs=dict(
        learning_rate=0.001
    ),
)
Wave_1_plus_1D = (
    WaveEquation2D,
    Wave_Config,
    Wave_Hyperparameters
)

# Kovasznay Flow
# ------------------------------------------------------
Kovasznay_N = (200,200)
Kovasznay_Config = dict()
Kovasznay_Hyperparameters = dict(
    n_steps=10000,
    ns=(Kovasznay_N,),
    n_test=Kovasznay_N,
    attention_tracking_kwargs=attention_params_1(reduce(mul, Kovasznay_N)),
    optimiser_kwargs=dict(
        learning_rate=0.0001
    ),
)
Kovasznay_Flow = (
    KovasznayFlow,
    Kovasznay_Config,
    Kovasznay_Hyperparameters
)


# Taylor Green Vortex
# ------------------------------------------------------
TaylorGreenVortex_N = (50,50,50)
TaylorGreenVortex_Config = dict()
TaylorGreenVortex_Hyperparameters = dict(
    n_steps=10000,
    ns=(TaylorGreenVortex_N,),
    n_test=TaylorGreenVortex_N,
    attention_tracking_kwargs=attention_params_1(reduce(mul, TaylorGreenVortex_N)),
    optimiser_kwargs=dict(
        learning_rate=0.001
    ),
    summary_freq = 100
)
TaylorGreenVortex = (
    TaylorGreen3DFlow,
    TaylorGreenVortex_Config,
    TaylorGreenVortex_Hyperparameters
)