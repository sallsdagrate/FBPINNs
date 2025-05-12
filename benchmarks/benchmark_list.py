from config.decompositions_config import Rectangle_1D_Decomposition, Rectangle_2D_Decomposition
from config.problems_config import HarmonicOscillator1D_LowFreq, HarmonicOscillator1D_HighFreq, Heat_Eq_1_plus_1D, Burgers_1_plus_1D, Poisson_2D
from config.domains_config import Rectangle_1D, Rectangle_2D


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

poisson_2d = {
    "Poisson_2D": {
        "problem": Poisson_2D,
        "domain": Rectangle_2D(0, 1, 0, 1),
        "decomposition": Rectangle_2D_Decomposition(0, 1, 0, 1, n1=5, n2=5),
        "dims": (2, 1),
    },
}