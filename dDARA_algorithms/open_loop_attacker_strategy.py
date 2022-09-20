import numpy as np
from blotto_prop import BlottoProp
from environments import Environment, generate_env_from_name
from utils.utils import generate_mesh_over_simplex, random_sample_over_simplex
from utils.utils import compute_x_req, req_2_simplex, isEqual
from convex_hull_algs import intersect
from copy import deepcopy


def open_loop_attacker_strategy(env: Environment, x_0, y_0, t_max, sampling_method="random", n_samples=None,
                                resolution=None):
    # Environment parameters
    X = env.X
    Y = env.Y

    prop_X = BlottoProp(connectivity=env.connectivity_X, agent_name="Defender")
    prop_Y = BlottoProp(connectivity=env.connectivity_Y, agent_name="Attacker")
    prop_Y_temp = BlottoProp(connectivity=env.connectivity_Y, agent_name="Attacker_temp")

    # Set up initial vertices
    prop_X.set_initial_vertices(initial_vertices=[x_0], perturb_singleton=False)
    prop_Y.set_initial_vertices(initial_vertices=[y_0], perturb_singleton=False)

    t = 1
    y_g = None
    intersection_flag = True

    while t < t_max and intersection_flag:
        R_x_vertices = prop_X.prop_step()
        R_y_vertices = prop_Y.prop_step()

        prop_X.append_flow(R_x_vertices)
        prop_Y.append_flow(R_y_vertices)

        # sample from previous step
        y_samples = []
        n_vertices = len(prop_Y.vertex_flow[t - 1])

        if t - 1 == 0:
            # previous step is the initial step
            y_samples.append(y_0)
        else:
            if sampling_method == "random":
                assert n_samples is not None
                for _ in range(n_samples):
                    # sample a convex coefficient
                    coefficient_sample = random_sample_over_simplex(x_dim=n_vertices, X=1)
                    y_sample = sum(
                        [prop_Y.vertex_flow[t - 1].vertices[i] * coefficient_sample[i] for i in range(n_vertices)])
                    y_samples.append(y_sample)
            elif sampling_method == "mesh":
                assert resolution is not None
                # create a mesh for the convex coefficients
                mesh = generate_mesh_over_simplex(resolution=resolution, x_dim=n_vertices, X=1)
                for coefficient_sample in mesh:
                    y_sample = sum(
                        [prop_Y.vertex_flow[t - 1].vertices[i] * coefficient_sample[i] for i in range(n_vertices)])
                    y_samples.append(y_sample)
            else:
                raise Exception("Sampling method not recognized!")

        for y_sample in y_samples:
            prop_Y_temp.set_initial_vertices(initial_vertices=[y_sample], perturb_singleton=False)
            R_y_vertices_temp = prop_Y_temp.prop_step()

            x_req = compute_x_req(R_y_vertices_temp)
            P_req = req_2_simplex(x_req=x_req, X=X)

            # check intersection of P_req and R_x
            _, _, intersection_flag = intersect(P_req, R_x_vertices, sloppy=False)
            if not intersection_flag:
                y_g = deepcopy(y_sample)
                break

        t += 1
    if intersection_flag:
        print("No solution found!")
    else:
        np.set_printoptions(precision=2)
        print("Solution found with y_0 = {}, y_g = {}".format(y_0, y_g))
    return y_g


if __name__ == "__main__":
    env_name = "figure-4-v2"
    env = generate_env_from_name(env_name)
    #               1  2  3  4  5  6  7
    x_0 = np.array([1, 0, 0, 0, 1, 1, 1])
    y_0 = np.array([1, 0, 0, 0, 0, 0, 0])

    assert isEqual(sum(x_0), env.X)
    assert isEqual(sum(y_0), env.Y)

    soln = open_loop_attacker_strategy(env=env, x_0=x_0, y_0=y_0, t_max=10, sampling_method="mesh", resolution=1)
