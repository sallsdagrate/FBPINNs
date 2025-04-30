from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.constants import get_subdomain_ws

import numpy as np

Rectangle_1D_Decomposition = lambda xmin, xmax, un=1, n=15, w=0.15: (
    RectangularDecompositionND,
    dict(
        subdomain_xs=[np.linspace(xmin,xmax,n)], 
        subdomain_ws=[w*(xmax - xmin)*np.ones((n,))], 
        unnorm=(0, un),
    )
)

Rectangle_2D_Decomposition = lambda xmin, xmax, ymin, ymax, un=1, n1=10, n2=10, w=2.9: (
    RectangularDecompositionND,
    dict(
        subdomain_xs=[np.linspace(xmin,xmax,n1), np.linspace(ymin,ymax,n2)], 
        subdomain_ws=get_subdomain_ws([np.linspace(xmin,xmax,n1), np.linspace(ymin,ymax,n2)], w), 
        unnorm=(0, un),
    )
)

