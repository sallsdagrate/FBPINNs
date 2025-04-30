from config.decompositions_config import Rectangle_1D_Decomposition, Rectangle_2D_Decomposition
from config.problems_config import HarmonicOscillator1D_LowFreq, HarmonicOscillator1D_HighFreq, Heat_Eq_1_plus_1D
from config.domains_config import Rectangle_1D, Rectangle_2D


benchmarks = {
    "HarmonicOscillator1D_LowFreq": {
        "problem": HarmonicOscillator1D_LowFreq,
        "domain": Rectangle_1D(0, 1),
        "decomposition": Rectangle_1D_Decomposition(0, 1),
        "dims": (1, 1),
    },
}

benchmarks_1 = {
    "HarmonicOscillator1D_HighFreq": {
        "problem": HarmonicOscillator1D_HighFreq,
        "domain": Rectangle_1D(0, 5),
        "decomposition": Rectangle_1D_Decomposition(0, 5),
        "dims": (1, 1),
    },
}

benchmarks_2 = {
    "Heat_Eq_1+1D": {
        "problem": Heat_Eq_1_plus_1D,
        "domain": Rectangle_2D(0, 1, 0, 1),
        "decomposition": Rectangle_2D_Decomposition(0, 1, 0, 1),
        "dims": (2, 1),
    },
}