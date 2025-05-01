from benchmarks.benchmark_list import burgers
from benchmarks.benchmarker import Benchmarker

from config.models_config import get_all_models, get_all_models_1
from fbpinns.trainers import FBPINNTrainer

models = get_all_models_1()
trainer = FBPINNTrainer
to_run = burgers

if __name__ == "__main__":
    # Create a Benchmarker instance
    print(f"Running benchmarks\n{to_run}")
    bench = Benchmarker(to_run, models, trainer)

    # Run the benchmarks
    results = bench.run()