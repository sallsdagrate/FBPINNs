from config.constant_config import Rectangle_1D_0_1_Decomposed
from config.problem_config import HarmonicOscillator1D_LowFreq
from config.model_config import get_all_models
from fbpinns.trainers import FBPINNTrainer
from benchmarks.benchmarker import Benchmarker


benchmarks = {
    "HarmonicOscillator1D_LowFreq": {
        "problem": HarmonicOscillator1D_LowFreq,
        "config": Rectangle_1D_0_1_Decomposed,
        "dims": (1, 1),
    },
}