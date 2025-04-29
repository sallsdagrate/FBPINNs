from benchmarks.benchmark_list import benchmarks
from benchmarks.benchmarker import Benchmarker

from config.models_config import get_all_models
from fbpinns.trainers import FBPINNTrainer

models = get_all_models()
trainer = FBPINNTrainer

if __name__ == "__main__":
    # Create a Benchmarker instance
    print(f"Running benchmarks\n{benchmarks}")
    bench = Benchmarker(benchmarks, models, trainer)

    # Run the benchmarks
    results = bench.run()