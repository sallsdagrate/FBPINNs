import numpy as np
from fbpinns.domains import RectangularDomainND
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.constants import Constants

# common between Constants
# ------------------------------------------------------
attention_config_1 = dict(
    eta_lr = 1e-2,
    gamma_decay = 0.99,
    out_dim=1,
    N=100
)

optimiser_config_1 = dict(
    learning_rate=0.001
)

# Constants
# ------------------------------------------------------
Rectangle_1D_0_1_Decomposed = Constants(
        domain=RectangularDomainND,
        domain_init_kwargs=dict(
            xmin=np.array([0,]),
            xmax=np.array([1,]),
        ),
        decomposition=RectangularDecompositionND,
        decomposition_init_kwargs=dict(
            subdomain_xs=[np.linspace(0,1,15)],
            subdomain_ws=[0.15*np.ones((15,))],
            unnorm=(0.,1.),
        ),
        optimiser_kwargs = optimiser_config_1,
        ns=((100,),),
        n_test=(100,),
        n_steps=10000,
        attention_tracking_kwargs=attention_config_1,
        save_figures=True,
        show_figures=False,
        clear_output=False,
    )