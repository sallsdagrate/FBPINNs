from fbpinns.decompositions import RectangularDecompositionND
import numpy as np

Rectangle_1D_Decomposition = lambda xmin, xmax, un=1: (
    RectangularDecompositionND,
    dict(
        subdomain_xs=[np.linspace(xmin,xmax,15)], 
        subdomain_ws=[0.15*(xmax - xmin)*np.ones((15,))], 
        unnorm=(0, un),
    )
)
