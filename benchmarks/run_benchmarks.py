from benchmarks.benchmark_list import benchmarks
from benchmarks.benchmarker import Benchmarker

from config.model_config import get_all_models
from fbpinns.trainers import FBPINNTrainer

models = get_all_models()
trainer = FBPINNTrainer

if __name__ == "__main__":
    # Initialize lists to store all labels, problems, configs, and dimensions
    all_labels = all_problems = all_configs = all_dims = []
    for benchmark_name, benchmark in benchmarks.items():
        all_labels.append(benchmark_name)
        all_problems.append(benchmark["problem"])
        all_configs.append(benchmark["config"])
        all_dims.append(benchmark["dims"])
    
    # Create a Benchmarker instance
    print(f"Running benchmarks")
    bench = Benchmarker(all_labels, all_problems, all_configs, models, trainer, all_dims)

    # Run the benchmarks
    results = bench.run()