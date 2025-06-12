from config.decompositions_config import Rectangle_1D_Decomposition, Rectangle_2D_Decomposition, Rectangle_3D_Decomposition
from config.problems_config import (
    HarmonicOscillator1D_LowFreq,
    HarmonicOscillator1D_HighFreq,
    Heat_Eq_1_plus_1D,
    Burgers_1_plus_1D,
    Poisson_2D,
    Schrodinger1D_Stationary,
    Schrodinger1D_NonStationary,
    Wave_1_plus_1D,
    Kovasznay_Flow,
    TaylorGreenVortex,
    WaveGaussianVelocity,
    Wave_1_plus_1D_High_50,
    Wave_1_plus_1D_High_100,    
    Wave_1_plus_1D_High_200,    
    Wave_1_plus_1D_High_400,
    Wave_1_plus_1D_grid,
    Wave_1_plus_1D_uniform,
    Wave_1_plus_1D_sobol,
    Wave_1_plus_1D_halton,
    Wave_1_plus_1D_Attention,
    Burgers_1_plus_1D_Attention
    )
from config.domains_config import Rectangle_1D, Rectangle_2D, Rectangle_3D
import jax.numpy as jnp


low_freq = {
    "HarmonicOscillator1D_LowFreq": {
        "problem": HarmonicOscillator1D_LowFreq,
        "domain": Rectangle_1D(0, 1),
        "decomposition": Rectangle_1D_Decomposition(0, 1),
        "dims": (1, 1),
    },
}

high_freq = {
    "HarmonicOscillator1D_HighFreq": {
        "problem": HarmonicOscillator1D_HighFreq,
        "domain": Rectangle_1D(0, 5),
        "decomposition": Rectangle_1D_Decomposition(0, 5),
        "dims": (1, 1),
    },
}

heat = {
    "Heat_Eq_1+1D": {
        "problem": Heat_Eq_1_plus_1D,
        "domain": Rectangle_2D(0, 1, 0, 1),
        "decomposition": Rectangle_2D_Decomposition(0, 1, 0, 1),
        "dims": (2, 1),
    },
}

burgers = {
    "Burgers_1+1D": {
        "problem": Burgers_1_plus_1D,
        "domain": Rectangle_2D(-1, 1, 0, 1),
        "decomposition": Rectangle_2D_Decomposition(-1, 1, 0, 1, n1=20, n2=10),
        "dims": (2, 1),
    },
}
burgers_attn = {
    "Burgers_1+1D_Attention": {
        "problem": Burgers_1_plus_1D_Attention,
        "domain": Rectangle_2D(-1, 1, 0, 1),
        "decomposition": Rectangle_2D_Decomposition(-1, 1, 0, 1, n1=20, n2=10),
        "dims": (2, 1),
    },
}

poisson_2d = {
    "Poisson_2D": {
        "problem": Poisson_2D,
        "domain": Rectangle_2D(0, 1, 0, 1),
        "decomposition": Rectangle_2D_Decomposition(0, 1, 0, 1, n1=5, n2=5),
        "dims": (2, 1),
    },
}

schrodinger_stationary = {
    "Schrodinger1D_Stationary": {
        "problem": Schrodinger1D_Stationary,
        "domain": Rectangle_2D(-5., 5., 0., jnp.pi),
        # "decomposition": Rectangle_2D_Decomposition(-5., 5., 0., jnp.pi, n1=20, n2=5),
        "decomposition": Rectangle_2D_Decomposition(-5., 5., 0., jnp.pi, n1=10, n2=5),
        "dims": (2, 2),
    },
}

schrodinger_non_stationary = {
    "Schrodinger1D_NonStationary": {
        "problem": Schrodinger1D_NonStationary,
        "domain": Rectangle_2D(-5., 5., 0., jnp.pi/2),
        "decomposition": Rectangle_2D_Decomposition(-5., 5., 0., jnp.pi/2, n1=20, n2=5),
        "dims": (2, 2),
    },
}

wave = {
    "Wave": {
        "problem": Wave_1_plus_1D,
        "domain": Rectangle_2D(0., 1., 0., 1.),
        "decomposition": Rectangle_2D_Decomposition(0., 1., 0., 1., n1=5, n2=5),
        "dims": (2, 1)
    }
}

kovasznay = {
    "Kovasznay": {
        "problem": Kovasznay_Flow,
        "domain": Rectangle_2D(0., 1., 0., 1.),
        "decomposition": Rectangle_2D_Decomposition(0., 1., 0., 1., n1=5, n2=5),
        "dims": (2, 3)
    }
}

taylor_pos = (0., 1., 0., 1., 0., 1.)
taylorgreen = {
    "taylorgreen": {
        "problem": TaylorGreenVortex,
        "domain": Rectangle_3D(*taylor_pos),
        "decomposition": Rectangle_3D_Decomposition(*taylor_pos),
        "dims": (3, 4)
    }
}

wave_eq_gauss_pos = (0., 1., 0., 1., 0., 1.)
wave_eq_gauss = {
    "wave_eq_gauss": {
        "problem": WaveGaussianVelocity,
        "domain": Rectangle_3D(*wave_eq_gauss_pos),
        "decomposition": Rectangle_3D_Decomposition(*wave_eq_gauss_pos),
        "dims": (3, 1)
    }
}


schrodinger_stationary_4_2 = {
    "Schrodinger1D_Stationary_4_2": {
        "problem": Schrodinger1D_Stationary,
        "domain": Rectangle_2D(-5., 5., 0., jnp.pi),
        "decomposition": Rectangle_2D_Decomposition(-5., 5., 0., jnp.pi, n1=5, n2=2),
        "dims": (2, 2),
    },
}
schrodinger_stationary_10_4 = {
    "Schrodinger1D_Stationary_10_4": {
        "problem": Schrodinger1D_Stationary,
        "domain": Rectangle_2D(-5., 5., 0., jnp.pi),
        "decomposition": Rectangle_2D_Decomposition(-5., 5., 0., jnp.pi, n1=10, n2=4),
        "dims": (2, 2),
    },
}
schrodinger_stationary_20_8 = {
    "Schrodinger1D_Stationary_20_8": {
        "problem": Schrodinger1D_Stationary,
        "domain": Rectangle_2D(-5., 5., 0., jnp.pi),
        "decomposition": Rectangle_2D_Decomposition(-5., 5., 0., jnp.pi, n1=20, n2=8),
        "dims": (2, 2),
    },
}

wave_high_n = {
    # "Wave_50": {
    #     "problem": Wave_1_plus_1D_High_50,
    #     "domain": Rectangle_2D(0., 1., 0., 1.),
    #     "decomposition": Rectangle_2D_Decomposition(0., 1., 0., 1., n1=5, n2=5),
    #     "dims": (2, 1)
    # },
    # "Wave_100": {
    #     "problem": Wave_1_plus_1D_High_100,
    #     "domain": Rectangle_2D(0., 1., 0., 1.),
    #     "decomposition": Rectangle_2D_Decomposition(0., 1., 0., 1., n1=5, n2=5),
    #     "dims": (2, 1)
    # },
    "Wave_200": {
        "problem": Wave_1_plus_1D_High_200,
        "domain": Rectangle_2D(0., 1., 0., 1.),
        "decomposition": Rectangle_2D_Decomposition(0., 1., 0., 1., n1=5, n2=5),
        "dims": (2, 1)
    },
    # "Wave_400": {
    #     "problem": Wave_1_plus_1D_High_400,
    #     "domain": Rectangle_2D(0., 1., 0., 1.),
    #     "decomposition": Rectangle_2D_Decomposition(0., 1., 0., 1., n1=5, n2=5),
    #     "dims": (2, 1)
    # }
}

wave_sampler = {
    # "Wave_grid": {
    #     "problem": Wave_1_plus_1D_grid,
    #     "domain": Rectangle_2D(0., 1., 0., 1.),
    #     "decomposition": Rectangle_2D_Decomposition(0., 1., 0., 1., n1=5, n2=5),
    #     "dims": (2, 1)
    # },
    "Wave_uniform": {
        "problem": Wave_1_plus_1D_uniform,
        "domain": Rectangle_2D(0., 1., 0., 1.),
        "decomposition": Rectangle_2D_Decomposition(0., 1., 0., 1., n1=5, n2=5),
        "dims": (2, 1)
    },
    "Wave_sobol": {
        "problem": Wave_1_plus_1D_sobol,
        "domain": Rectangle_2D(0., 1., 0., 1.),
        "decomposition": Rectangle_2D_Decomposition(0., 1., 0., 1., n1=5, n2=5),
        "dims": (2, 1)
    },
    "Wave_halton": {
        "problem": Wave_1_plus_1D_halton,
        "domain": Rectangle_2D(0., 1., 0., 1.),
        "decomposition": Rectangle_2D_Decomposition(0., 1., 0., 1., n1=5, n2=5),
        "dims": (2, 1)
    }
}

wave_attention = {
    "Wave_Attention": {
        "problem": Wave_1_plus_1D_Attention,
        "domain": Rectangle_2D(0., 1., 0., 1.),
        "decomposition": Rectangle_2D_Decomposition(0., 1., 0., 1., n1=5, n2=5),
        "dims": (2, 1)
    }
}