import numpy as np
from blotto_prop import BlottoProp
import matplotlib.pyplot as plt
from utils import compute_y_req_v1, compute_y_req_v2, compute_x_req, compare_vertices
from copy import deepcopy


x_0 = np.array([0.1, 0.3, 0.6])

T = 2 # Terminal time

Connectivity = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
N = len(Connectivity)

prop_X = BlottoProp(connectivity=Connectivity, x0=x_0, T=T, agent_name="")

ax_X = prop_X.plot_simplex(t=0)
ax_X = prop_X.vertex_flow[-1].plot(ax_X, color='m', legend='initial state')
plt.show()

for t in range(1, T + 1):

    # propogate feasible region
    x_vertices = prop_X.prop_step()
    ax_X = prop_X.plot_simplex(t=t)
    ax_X = x_vertices.plot(ax_X, color='m', legend='feasible region')
    plt.show()

    prop_X.append_flow(x_vertices)