import numpy as np
from fbpinns.domains import RectangularDomainND


Rectangle_1D = lambda xmin, xmax: (
    RectangularDomainND,
    dict(
        xmin=np.array([xmin,]),
        xmax=np.array([xmax,]),
    ),
)

Rectangle_2D = lambda xmin, xmax, ymin, ymax: (
    RectangularDomainND,
    dict(
        xmin=np.array([xmin,ymin]),
        xmax=np.array([xmax,ymax]),
    ),
)