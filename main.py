import numpy as np
from blotto_prop import BlottoProp
import matplotlib.pyplot as plt
from utils import compute_y_req_v1, compute_y_req_v2, compute_x_req, compare_vertices
from copy import deepcopy

x_0 = 2 * np.array([0.1, 0.3, 0.6])
y_0 = np.array([0.7, 0.1, 0.2])

T = 5  # Terminal time

Connectivity = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
N = len(Connectivity)

prop_X = BlottoProp(connectivity=Connectivity, x0=x_0, T=T, agent_name="Defender", hull_method='aux_point',
                    need_connections=False)
prop_Y = BlottoProp(connectivity=Connectivity, x0=y_0, T=T, agent_name="Attacker")

# plot agents' initial distributions
ax_Y = prop_Y.plot_simplex(t=0)
ax_Y = prop_Y.vertex_flow[-1].plot(ax_Y, color='m', legend='Attacker')

ax_X = prop_X.plot_simplex(t=0)
ax_X = prop_X.vertex_flow[-1].plot(ax_X, color='m', legend='Defender')

for t in range(1, T + 1):

    # propogate feasible region
    x_vertices = prop_X.prop_step()

    y_vertices = prop_Y.prop_step()

    ax_X = prop_X.plot_simplex(t=0)
    ax_X = x_vertices.plot(ax_X, color='m', legend='Defender')
    plt.show()

    x_cutted_vertices = deepcopy(x_vertices)

    # start the triangular iteration
    done_flag = False
    while not done_flag:
        # cut y region based on current x_vertices at t
        y_req = compute_y_req_v2(x_cutted_vertices, eta=0.2)
        y_cut_vertices = prop_Y.req_2_simplex(y_req)
        y_cutted_vertices = prop_Y.cut(y_cut_vertices, y_vertices)

        # append cutted vertices to the flow
        prop_Y.append_flow(y_cutted_vertices)

        # prop y region to t+1
        y_vertices_next = prop_Y.prop_step()

        # remove the y_cut added
        prop_Y.revert_step()

        # compute feasible region of X based on y_next
        x_req = compute_x_req(y_vertices_next)

        # Normalize x_req
        x_req /= x_y_ratio

        if sum(x_req) > 1:
            raise Exception("Feasible region empty for defender")

        x_cut_vertices = prop_X.req_2_simplex(x_req)
        x_cutted_vertices_new = prop_X.cut(x_cut_vertices, x_vertices)

        if compare_vertices(x_cutted_vertices, x_cutted_vertices_new):
            done_flag = True
            prop_X.append_flow(x_cutted_vertices_new)
            prop_Y.append_flow(y_cutted_vertices)
        else:
            x_cutted_vertices = x_cutted_vertices_new

    # plot Y's feasible region
    ax_Y = prop_Y.plot_simplex(t=t)
    ax_Y = y_vertices.plot(ax_Y, color='m', legend='original feasible region')
    ax_Y = y_cut_vertices.plot(ax_Y, color='y', legend='y_req')
    ax_Y = y_cutted_vertices.plot(ax_Y, color='r', legend='cutted')
    plt.show()

    # plot X's feasible region
    ax_X = prop_X.plot_simplex(t=t)
    ax_X = x_vertices.plot(ax_X, color='m', legend='original feasible region')
    ax_X = x_cut_vertices.plot(ax_X, color='y', legend='x_req')
    ax_X = x_cutted_vertices.plot(ax_X, color='r', legend='cutted')
    plt.show()

    print("Time step {} finished!".format(t))
