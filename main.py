import numpy as np
from blotto_prop import BlottoProp
import matplotlib.pyplot as plt

x_0 = np.array([0.1, 0.3, 0.6])
y_0 = np.array([0.3, 0.5, 0.2])

x_y_ratio = 2.8 # ratio of x/y

T = 5  # Terminal time

Connectivity = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
N = len(Connectivity)

prop_X = BlottoProp(connectivity=Connectivity, x0=x_0, T=T, agent_name="X")
prop_Y = BlottoProp(connectivity=Connectivity, x0=y_0, T=T, agent_name="Y")

for t in range(1, T + 1):

    vertices_x = prop_X.prop_step()
    vertices_y = prop_Y.prop_step()

    ax_Y = prop_Y.plot_simplex(t=t)
    ax_Y = prop_X.vertex_flow[-1].plot(ax_Y, color='m', legend='feasible region')

    ax_X = prop_X.plot_simplex(t=t)
    ax_X = prop_X.vertex_flow[-1].plot(ax_X, color='m', legend='original feasible region')

    x_req = np.zeros(N)
    for i in range(N):
        vertices_y_i = [vertices_y[k][i] for k in range(len(vertices_y))]
        x_req[i] = max(vertices_y_i)

    x_req /= x_y_ratio

    cut_vertices = prop_X.x_req_2_simplex(x_req)
    ax_X = cut_vertices.plot(ax_X, color='y', legend='x_req')

    cutted_vertices = prop_X.cut(cut_vertices)
    ax_X = cutted_vertices.plot(ax_X, color='r', legend='cutted')

    plt.show()



print("done")
