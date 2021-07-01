import numpy as np
from blotto_prop import BlottoProp

x_0 = np.array([0.1, 0.3, 0.6])


Connectivity = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
N = len(Connectivity)

prop = BlottoProp(connectivity=Connectivity, x0=x_0, T=3)
prop.prop()
