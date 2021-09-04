import numpy as np
from blotto_prop import BlottoProp
from convex_hull_algs import isInHull
from copy import deepcopy


# figure 4 scenario in paper
Connectivity = np.array([[1, 1, 0, 0, 1],
                         [0, 1, 1, 1, 0],
                         [0, 0, 1, 1, 0],
                         [0, 0, 1, 1, 1],
                         [1, 0, 0, 0, 1]])

x_s = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
x_g = np.array([0.0, 0.0, 0.0, 1.0, 0.0])

T_max = 10

prop_X = BlottoProp(connectivity=Connectivity, x0=x_s, T=T_max, agent_name="Attacker", hull_method="aux_point",
                    need_connections=False)

tau = 0

for t in range(T_max + 1):
    tau += 1
    new_vertices = prop_X.prop_step()
    if isInHull(x_g, deepcopy(new_vertices.vertices)):
        print(tau)
        break

    prop_X.append_flow(new_vertices)

