import numpy as np
from config.config_builder import ConfigBuilder
import subprocess
import json


class Benchmarker:
    def __init__(self, benchmarks, models, trainer):
        self.benchmarks = benchmarks
        self.models = models
        self.trainer = trainer
        self.results = []

    def run(self):
        # empty list to store results
        self.results = []

        # Loop through all problems, configs and models
        for label, config_args in self.benchmarks.items():
            result = self.run_problem(label, config_args)
            self.results.append(result)
        return self.results

    # runs problem with one config for all models
    def run_problem(self, label, config_args):
        assert len(self.models) > 0, "No models to run"

        # initialise models for the problem with the given dimensions
        indim, outdim = config_args['dims']
        constructed_models = [model(indim, outdim) for model in self.models]

        return [self.run_problem_with_model(label, config_args, model) for model in constructed_models]

    # runs problem with one config for one model
    def run_problem_with_model(self, label, config_args, model):
        model_name, model, model_args = model
        problem, problem_args, problem_hyperparams = config_args['problem']
        domain, domain_args = config_args['domain']
        decomposition, decomposition_args = config_args['decomposition']

        # build configuration for specified problem
        config = ConfigBuilder()\
            .with_problem(problem, problem_args, problem_hyperparams)\
            .with_domain(domain, domain_args)\
            .with_decomposition(decomposition, decomposition_args)\
            .with_network(model, model_args)\
            .build_FBPINN_config()

        # run the trainer
        run = self.trainer(config)
        result, metrics = run.train()
        
        # save the results
        np.savez("results/saved_arrays/test_mse.npz", mse=metrics['mse'])
        # delete mse from metrics after saving so that other metrics can be saved in json
        del metrics['mse']
        with open("results/saved_arrays/test_meta.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # run cleanup script to move results into self-contained folder
        self.cleanup_run(label, model_name)
        
        return label, config, model, result, metrics
    
    def cleanup_run(self, problem_name, model_name):
        subprocess.run(["./benchmarks/benchmark_collect.sh", problem_name, model_name])


    
# if __name__ == "__main__":
#     # Example usage
#     labels = ["HarmonicOscillator1D_LowFreq"]
#     problems = [HarmonicOscillator1D_LowFreq]
#     configs = [Rectangle_1D_0_1_Decomposed]
#     models = [FCN_Generator(8)]
#     trainer = FBPINNTrainer
#     dummy_dims = [(1, 1)]

#     benchmarker = Benchmarker(labels, problems, configs, models, trainer, dummy_dims)
#     results = benchmarker.run()
#     # print(results)