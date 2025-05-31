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
    get_models_stacked_lkan_varying_hidden_dims,
    get_models_optimized_ckan,
    get_models_optimized_stacked_ckan,
    get_models_taylor,
    get_models_mixed_basis_kan,
    get_models_scheduling,
    )

from fbpinns.trainers import FBPINNTrainer

#get_all_models_1() + get_all_models_2()
#  + get_models_stacked_ckan_varying_stack() + get_models_stacked_ckan_varying_hidden_dims() + get_models_stacked_lkan_varying_hidden_dims()
#get_models_optimized_ckan() + get_models_optimized_stacked_ckan()
models =  get_models_scheduling()
trainer = FBPINNTrainer
to_run = kovasznay | poisson_2d | schrodinger_stationary | wave
schedule = True

if __name__ == "__main__":
    # Create a Benchmarker instance
    print(f"Running benchmarks\n{to_run}")
    bench = Benchmarker(to_run, models, trainer, scheduled=schedule)

    # Run the benchmarks
    results = bench.run()