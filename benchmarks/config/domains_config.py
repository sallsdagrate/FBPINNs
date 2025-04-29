import numpy as np
from fbpinns.domains import RectangularDomainND


Rectangle_1D = lambda xmin, xmax: (
    RectangularDomainND,
    dict(
        xmin=np.array([xmin,]),
        xmax=np.array([xmax,]),
    ),
)