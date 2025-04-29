from fbpinns.problems import HarmonicOscillator1D

# Problem Configurations
# ------------------------------------------------------
HarmonicOscillator1D_N = 500
HarmonicOscillator1D_Config = dict(d=4, w0=80)
HarmonicOscillator1D_Hyperparameters = dict(
    n_steps=10000,
    n_test=(HarmonicOscillator1D_N,),
    ns=((HarmonicOscillator1D_N,),),
    attention_tracking_kwargs=dict(
        eta_lr=1e-2,
        gamma_decay=0.99,
        out_dim=1,
        N=HarmonicOscillator1D_N
    ),
    optimiser_kwargs=dict(
        learning_rate=0.001
    ),
)
HarmonicOscillator1D_LowFreq = (HarmonicOscillator1D, 
                                HarmonicOscillator1D_Config,
                                HarmonicOscillator1D_Hyperparameters)