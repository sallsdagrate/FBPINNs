from config.decompositions_config import Rectangle_1D_Decomposition
from config.problems_config import HarmonicOscillator1D_LowFreq
from config.domains_config import Rectangle_1D


benchmarks = {
    "HarmonicOscillator1D_LowFreq_wwew": {
        "problem": HarmonicOscillator1D_LowFreq,
        "domain": Rectangle_1D(0, 1),
        "decomposition": Rectangle_1D_Decomposition(0, 1),
        "dims": (1, 1),
    },
}