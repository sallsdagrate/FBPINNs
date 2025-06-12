from benchmarks.benchmark_list import *
from benchmarks.benchmarker import Benchmarker

from config.models_config import *

from fbpinns.trainers import FBPINNTrainer

#get_all_models_1() + get_all_models_2()
#  + get_models_stacked_ckan_varying_stack() + get_models_stacked_ckan_varying_hidden_dims() + get_models_stacked_lkan_varying_hidden_dims()
#get_models_optimized_ckan() + get_models_optimized_stacked_ckan()
models =  get_models_polynomials()
trainer = FBPINNTrainer
to_run = burgers_attn#burgers | 
schedule = False

if __name__ == "__main__":
    # Create a Benchmarker instance
    print(f"Running benchmarks\n{to_run}")
    bench = Benchmarker(to_run, models, trainer, scheduled=schedule)

    # Run the benchmarks
    results = bench.run()