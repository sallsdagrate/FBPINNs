from benchmarks.benchmark_list import *
from benchmarks.benchmarker import Benchmarker

from config.models_config import (
    get_all_models, 
    get_all_models_1, 
    get_all_models_2, 
    get_models_stacked_ckan_varying_stack, 
    get_models_stacked_ckan_varying_hidden_dims, 
    get_models_stacked_ckan_varying_degrees_1, 
    get_models_stacked_ckan_varying_degrees_2,
    get_models_stacked_lkan_varying_hidden_dims
    )

from fbpinns.trainers import FBPINNTrainer
#get_all_models_1() + get_all_models_2()
models = get_models_stacked_ckan_varying_stack() + get_models_stacked_ckan_varying_hidden_dims() + get_models_stacked_lkan_varying_hidden_dims()
trainer = FBPINNTrainer
to_run = schrodinger_non_stationary | wave | burgers

if __name__ == "__main__":
    # Create a Benchmarker instance
    print(f"Running benchmarks\n{to_run}")
    bench = Benchmarker(to_run, models, trainer)

    # Run the benchmarks`
    results = bench.run()