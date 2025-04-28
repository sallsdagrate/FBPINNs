import numpy as np
from fbpinns.trainers import FBPINNTrainer
from config.constant_config import Rectangle_1D_0_1_Decomposed
from config.problem_config import HarmonicOscillator1D_LowFreq
from config.model_config import FCN_8
import subprocess
import json


class Benchmarker:
    def __init__(self, labels, problems, configs, models, trainer, dims):
        assert len(problems) == len(configs), "Number of problems and configs must match"
        
        self.labels = labels
        self.problems = problems
        self.configs = configs
        self.trainer = trainer
        self.dims = dims
        self.models = models
        self.results = []

    def run(self):
        # Loop through all problems, configs and models
        self.results = []
        for label, problem, config, dims in zip(self.labels, self.problems, self.configs, self.dims):
            result = self.run_problem(label, problem, config, dims)
            self.results.append(result)
        return self.results

    # runs problem with one config for all models
    def run_problem(self, label, problem, config, dims):
        problem_class, problem_kwargs = problem
        config.problem = problem_class
        config.problem_init_kwargs = problem_kwargs
        indim, outdim = dims
        constructed_models = [model(indim, outdim) for model in self.models]

        return [self.run_problem_with_model(label, problem, config, model) for model in constructed_models]

    # runs problem with one config for one model
    def run_problem_with_model(self, label, problem, config, model):
        assert len(self.models) > 0, "No models to run"

        # get insert model into the config
        _, model_class, model_kwargs = model
        config.network = model_class
        config.network_init_kwargs = model_kwargs

        # run the trainer
        run = self.trainer(config)
        result, metrics = run.train()
        
        # save the result
        np.savez("results/saved_arrays/test_mse.npz", mse=metrics['mse'])
        del metrics['mse']

        with open("results/saved_arrays/test_meta.json", "w") as f:
            json.dump(metrics, f, indent=4)

        self.cleanup_run(label, model[0])
        
        return label, problem, config, model, result, metrics
    
    def cleanup_run(self, problem_name, model_name):
        subprocess.run(["./benchmarks/benchmark_collect.sh", problem_name, model_name])


    
if __name__ == "__main__":
    # Example usage
    labels = ["HarmonicOscillator1D_LowFreq"]
    problems = [HarmonicOscillator1D_LowFreq]
    configs = [Rectangle_1D_0_1_Decomposed]
    models = [FCN_8]
    trainer = FBPINNTrainer
    dummy_dims = [(1, 1)]

    benchmarker = Benchmarker(labels, problems, configs, models, trainer, dummy_dims)
    results = benchmarker.run()
    # print(results)