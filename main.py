import numpy as np
from blotto_prop import BlottoProp
import matplotlib.pyplot as plt
from copy import deepcopy
from environments import generate_env_from_name, Environment
from utils import generate_x_req_set, Vertices
from convex_hull_algs import intersect


def blotto_prop_cut(env: Environment, x_0, y_0, Tf, visualize=False):
    need_connection = True if visualize else False

    # set up graph info
    connectivity_X = env.connectivity_X
    connectivity_Y = env.connectivity_Y

    # set up propagation algorithm
    prop_X = BlottoProp(connectivity=connectivity_X, T=Tf, agent_name='Defender', hull_method='aux_point',
                        need_connections=need_connection)
    prop_X.set_initial_vertices([x_0])

    prop_Y = BlottoProp(connectivity=connectivity_Y, T=Tf, agent_name='Attacker', hull_method='aux_point',
                        need_connections=need_connection)
    prop_Y.set_initial_vertices([y_0])

    # record set flow
    x_req_flow = [generate_x_req_set(vertices_y=Vertices([y_0]), X=env.X)]
    x_reachable_set_flow = [Vertices([x_0])]
    x_safe_set_flow = [Vertices([x_0])]

    y_reachable_set_flow = [Vertices([y_0])]

    for t in range(1, Tf + 1):
        # propagate feasible region
        x_vertices_t = prop_X.prop_step()
        y_vertices_t = prop_Y.prop_step()

        # cut x_vertices_t
        x_vertices_t_tmp = deepcopy(x_vertices_t)
        x_vertices_req_t = generate_x_req_set(vertices_y=y_vertices_t, X=env.X)
        x_safe_vertices_t, _, found = intersect(vertices1=x_vertices_req_t, vertices2=x_vertices_t_tmp, sloppy=False,
                                                need_connections=True)

        # record vertices
        x_reachable_set_flow.append(x_vertices_t)
        y_reachable_set_flow.append(y_vertices_t)
        x_req_flow.append(x_vertices_req_t)
        x_safe_set_flow.append(x_safe_vertices_t)

        # append vertices
        prop_X.append_flow(x_safe_vertices_t)
        prop_Y.append_flow(y_vertices_t)

        # break if the intersection is empty
        if not found:
            print('At time step {}, the Defender safe set is empty'.format(t))
            break

    if visualize:
        n_time_steps = len(prop_Y.vertex_flow)
        fig_X = plt.figure(figsize=(6 * n_time_steps, 6), dpi=120)
        fig_Y = plt.figure(figsize=(6 * n_time_steps, 6), dpi=120)
        for t in range(n_time_steps):
            ax_X = fig_X.add_subplot(1, n_time_steps, t + 1, projection='3d')
            prop_X.plot_simplex(t=t, ax=ax_X, title=False)
            x_reachable_set_flow[t].plot(ax=ax_X, color='m', legend='reachable region')
            x_req_flow[t].plot(ax=ax_X, color='y', legend='required resource')
            x_safe_set_flow[t].plot(ax=ax_X, color='r', legend='safe region')

            ax_Y = fig_Y.add_subplot(1, n_time_steps, t + 1, projection='3d')
            prop_Y.plot_simplex(t=t, ax=ax_Y, title=False)
            y_reachable_set_flow[t].plot(ax=ax_Y, color='m', legend='reachable region')

        fig_X.suptitle('Defender Reachable Set Analysis')
        fig_Y.suptitle('Attacker Reachable Set Analysis')

        plt.show()


if __name__ == "__main__":
    # Initial state
    x_0 = np.array([0.9, 0.4, 1.8])
    y_0 = np.array([0.3, 0.4, 0.3])

    # Terminal time step
    Tf = 5

    env = generate_env_from_name('3-node')
    blotto_prop_cut(x_0=x_0, y_0=y_0, env=env, Tf=Tf, visualize=True)
