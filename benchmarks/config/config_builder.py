from fbpinns.constants import Constants

class ConfigBuilder:

    def __init__(self):
        self.config = dict(
            save_figures=True,
            show_figures=False,
            clear_output=False,
            )
        self.has_problem = False
        self.has_domain = False
        self.has_decomposition = False
        self.has_network = False

    def __build__(self):
        ret = Constants(**self.config)
        self.__init__()
        return ret
    
    def build_PINN_config(self):
        if not self.has_problem:
            raise ValueError("Problem not set")
        if not self.has_domain:
            raise ValueError("Domain not set")
        if not self.has_network:
            raise ValueError("Network not set")
        return self.__build__()
    
    def build_FBPINN_config(self):
        if not self.has_decomposition:
            raise ValueError("Decomposition not set")
        return self.build_PINN_config()

    def with_domain(self, domain, domain_kwargs):
        self.config['domain'] = domain
        self.config['domain_init_kwargs'] = domain_kwargs
        self.has_domain = True
        return self
    
    def with_decomposition(self, decomposition, decomposition_kwargs):
        self.config['decomposition'] = decomposition
        self.config['decomposition_init_kwargs'] = decomposition_kwargs
        self.has_decomposition = True
        return self
    
    def with_problem(self, problem, problem_kwargs, problem_hyperparams=None):
        self.config['problem'] = problem
        self.config['problem_init_kwargs'] = problem_kwargs
        if problem_hyperparams is not None:
            self.config = self.config | problem_hyperparams
        self.has_problem = True
        return self
    
    def with_network(self, network, network_kwargs):
        self.config['network'] = network
        self.config['network_init_kwargs'] = network_kwargs
        self.has_network = True
        return self
    
    def with_attention(self, attention_tracking_kwargs):
        self.config['attention_tracking_kwargs'] = attention_tracking_kwargs
        return self
    
    def with_optimiser(self, optimiser_kwargs):
        self.config['optimiser_init_kwargs'] = optimiser_kwargs
        return self
    
    def with_scheduler(self, scheduler):
        self.config['scheduler'] = scheduler
        return self
    
    def print_state(self):
        print("Current state of the config builder:")
        for key, value in self.config.items():
            print(f"{key}: {value}")
        return self
    

    
    
