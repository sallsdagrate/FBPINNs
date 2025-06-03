from fbpinns.constants import Constants
from fbpinns.schedulers import LineSchedulerRectangularND

scheduled = lambda n: dict(
    scheduler = LineSchedulerRectangularND,
    scheduler_kwargs = dict(
        point=[0.] * n, iaxis=0,
    )
)

class ConfigBuilder:
    """
    Builder for creating configuration objects for PINN and FBPINN workflows.
    """

    def __init__(self):
        # Initialize default configuration and flags for required components
        self.config = dict(
            save_figures=True,   # whether to save training/validation figures
            show_figures=False,  # whether to display figures interactively
            clear_output=False,  # whether to clear console output between epochs
        )
        # Flags to enforce required builder steps before finalization
        self.has_problem = False
        self.has_domain = False
        self.has_decomposition = False
        self.has_network = False

    def __build__(self):
        """
        Internal: Instantiate a Constants object from the accumulated config,
        then reset the builder for reuse.
        """
        ret = Constants(**self.config)
        # Reinitialize builder state after building
        self.__init__()
        return ret
    
    def build_PINN_config(self):
        """
        Build the configuration for a standard PINN.
        Raises if any of the required components (problem, domain, network) are not set.
        """
        if not self.has_problem:
            raise ValueError("Problem not set")
        if not self.has_domain:
            raise ValueError("Domain not set")
        if not self.has_network:
            raise ValueError("Network not set")
        # All required parts are present, build and return the Constants
        return self.__build__()
    
    def build_FBPINN_config(self):
        """
        Build the configuration for a FBPINN (domain-decomposed PINN).
        Ensures a decomposition has been specified, then delegates to build_PINN_config.
        """
        if not self.has_decomposition:
            raise ValueError("Decomposition not set")
        # Delegate to PINN build after verifying decomposition
        return self.build_PINN_config()

    def with_domain(self, domain, domain_kwargs):
        """
        Specify the spatial/temporal domain for the PINN.

        Args:
            domain: a Domain class or factory
            domain_kwargs: initialization parameters for the domain

        Returns:
            self: to allow method chaining
        """
        self.config['domain'] = domain
        self.config['domain_init_kwargs'] = domain_kwargs
        self.has_domain = True
        return self
    
    def with_scheduling(self, schedule_n, scheduling_kwargs=scheduled):
        """
        Specify scheduling parameters for the FBPINN.

        Args:
            schedule_n: dimension of the scheduling space
            scheduling_kwargs: dict of parameters for scheduling (e.g., LineSchedulerRectangularND)
        Returns:
            self: to allow method chaining
        """
        if schedule_n:
            self.config['scheduler'] = scheduling_kwargs(schedule_n)['scheduler']
            self.config['scheduler_kwargs'] = scheduling_kwargs(schedule_n)['scheduler_kwargs']
        return self
    
    def with_decomposition(self, decomposition, decomposition_kwargs):
        """
        Specify how the domain should be decomposed for FBPINN.

        Args:
            decomposition: a Decomposition class or factory
            decomposition_kwargs: initialization parameters for decomposition
        """
        self.config['decomposition'] = decomposition
        self.config['decomposition_init_kwargs'] = decomposition_kwargs
        self.has_decomposition = True
        return self
    
    def with_problem(self, problem, problem_kwargs, problem_hyperparams=None):
        """
        Define the PDE problem to solve.

        Args:
            problem: a Problem class or factory
            problem_kwargs: initialization parameters for the problem
            problem_hyperparams: optional dict of hyperparameters (e.g., loss weights)
        """
        self.config['problem'] = problem
        self.config['problem_init_kwargs'] = problem_kwargs
        # Merge hyperparameters into main config if provided
        if problem_hyperparams is not None:
            self.config = self.config | problem_hyperparams
        self.has_problem = True
        return self
    
    def with_network(self, network, network_kwargs):
        """
        Configure the neural network architecture to use.

        Args:
            network: a Network class or factory
            network_kwargs: initialization parameters for the network
        """
        self.config['network'] = network
        self.config['network_init_kwargs'] = network_kwargs
        self.has_network = True
        return self
    
    def with_attention(self, attention_tracking_kwargs):
        """
        Optionally enable attention tracking within the network.

        Args:
            attention_tracking_kwargs: dict of parameters for tracking attention
        """
        self.config['attention_tracking_kwargs'] = attention_tracking_kwargs
        return self
    
    def with_optimiser(self, optimiser_kwargs):
        """
        Specify the optimiser settings for training.

        Args:
            optimiser_kwargs: initialization parameters for the optimiser
        """
        self.config['optimiser_init_kwargs'] = optimiser_kwargs
        return self
    
    def with_scheduler(self, scheduler):
        """
        Attach a learning-rate scheduler.

        Args:
            scheduler: a Scheduler class or factory
        """
        self.config['scheduler'] = scheduler
        return self
    
    def print_state(self):
        """
        Utility to print the current builder state and accumulated config values.
        Useful for debugging the builder chain.
        """
        print("Current state of the config builder:")
        for key, value in self.config.items():
            print(f"{key}: {value}")
        return self
