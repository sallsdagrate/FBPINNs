import numpy as np
import subprocess
import json
from config.config_builder import ConfigBuilder

class Benchmarker:
    """
    Orchestrates benchmarking of multiple models across different PDE problems.
    Iterates over given benchmarks, constructs configuration for each combination,
    trains models via a provided trainer, and collects results.
    """

    def __init__(self, benchmarks, models, trainer, scheduled=False):
        """
        Initialize the benchmarker.

        Args:
            benchmarks (dict): Mapping from label to configuration arguments,
                               including problem, domain, decomposition and dims.
            models (list):   List of model factory tuples (name, class, init_args).
            trainer (callable): Trainer class or factory that accepts a Constants config.
        """
        self.benchmarks = benchmarks    # store mapping of problem labels -> config args
        self.models = models            # store list of model constructors
        self.trainer = trainer          # trainer callable for running experiments
        self.results = []               # will hold results of benchmark runs
        self.scheduled = scheduled     # whether to run benchmarks with scheduling

    def run(self):
        """
        Execute all benchmarks in sequence.

        Returns:
            list: Results for each benchmark label, as returned by run_problem.
        """
        # Reset results list at start of run
        self.results = []

        # Loop through each benchmark setting and execute
        for label, config_args in self.benchmarks.items():
            result = self.run_problem(label, config_args)
            self.results.append(result)

        return self.results

    def run_problem(self, label, config_args):
        """
        Benchmark a single problem configuration across all provided models.

        Args:
            label (str):        Identifier for the problem instance.
            config_args (dict): Contains 'dims', 'problem', 'domain', 'decomposition'.

        Returns:
            list: Results for each model on this problem.
        """
        # Ensure at least one model is provided
        assert len(self.models) > 0, "No models to run"

        # Unpack input/output dimensions for model construction
        indim, outdim = config_args['dims']
        # Instantiate each model with corresponding dimensions
        constructed_models = [model(indim, outdim) for model in self.models]

        # Run this problem for each constructed model
        return [self.run_problem_with_model(label, config_args, mdl)
                for mdl in constructed_models]

    def run_problem_with_model(self, label, config_args, model):
        """
        Run a single model on a specific problem configuration.

        Args:
            label (str):        Problem identifier.
            config_args (dict): Contains problem, domain, decomposition tuples.
            model (tuple):      (model_name, model_class, model_init_kwargs)

        Returns:
            tuple: (label, config, model, training_result, metrics)
        """
        # Unpack model information
        model_name, model_cls, model_args = model
        # Unpack PDE problem specification
        problem, problem_args, problem_hyperparams = config_args['problem']
        # Unpack spatial/temporal domain
        domain, domain_args = config_args['domain']
        # Unpack domain decomposition for FBPINN
        decomposition, decomposition_args = config_args['decomposition']

        # Build a FBPINN configuration via the fluent ConfigBuilder
        config = (ConfigBuilder()
                  .with_problem(problem, problem_args, problem_hyperparams)
                  .with_domain(domain, domain_args)
                  .with_decomposition(decomposition, decomposition_args)
                  .with_network(model_cls, model_args)
                  .with_scheduling(self.scheduled)
                  .build_FBPINN_config())
        
        # Instantiate trainer and execute training
        run = self.trainer(config)
        result, metrics = run.train()

        # Persist MSE to a NumPy archive for numerical analysis
        np.savez("results/saved_arrays/test_mse.npz", mse=metrics['mse'])
        # Remove MSE from metrics dict so JSON file only holds other metadata
        del metrics['mse']
        # Save remaining metrics (e.g., runtime, loss history) to JSON
        with open("results/saved_arrays/test_meta.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Collect outputs into organized directory structure
        self.cleanup_run(label+('_scheduled' if self.scheduled else ''), model_name)

        return (label, config, model_name, result, metrics)

    def cleanup_run(self, problem_name, model_name):
        """
        Post-process and relocate benchmark artifacts via external script.

        Args:
            problem_name (str): Label of the problem instance.
            model_name (str):   Identifier of the model used.
        """
        # Invoke shell script to move logs, figures, and metrics
        subprocess.run(["./benchmarks/benchmark_collect.sh",
                        problem_name, model_name])
