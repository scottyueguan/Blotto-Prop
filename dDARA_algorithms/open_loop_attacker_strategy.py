import numpy as np
from blotto_prop import BlottoProp
from environments import Environment, generate_env_from_name
from utils import generate_mesh_over_simplex, random_sample_over_simplex
from utils import compute_x_req, req_2_simplex
from convex_hull_algs import intersect
from copy import deepcopy


def open_loop_attacker_strategy(env: Environment, x_0, y_0, t_max, n_samples=None, resolution=None,
                                sampling_method="random"):
    # Environment parameters
    X = env.X
    Y = env.Y

    prop_X = BlottoProp(connectivity=env.connectivity_X, agent_name="Defender")
    prop_Y = BlottoProp(connectivity=env.connectivity_Y, agent_name="Attacker")

    # Set up initial vertices
    prop_X.set_initial_vertices(initial_vertices=[x_0], perturb_singleton=True)
    prop_Y.set_initial_vertices(initial_vertices=[y_0], perturb_singleton=True)

    t = 1
    y_g = None

    while t < t_max:
        R_x_vertices = prop_X.prop_step()
        R_y_vertices = prop_Y.prop_step()

        prop_X.append_flow(R_x_vertices)
        prop_Y.append_flow(R_y_vertices)

        # sample from previous step
        y_samples = []
        n_vertices = len(prop_Y.vertex_flow[t - 1])

        if sampling_method == "random":
            for _ in range(n_samples):
                # sample a convex coefficient
                coefficient_sample = random_sample_over_simplex(x_dim=n_vertices, X=1)
                y_sample = sum(
                    [prop_Y.vertex_flow[t - 1].vertices[i] * coefficient_sample[i] for i in range(n_vertices)])
                y_samples.append(y_sample)
                if t - 1 == 0:
                    # previous step is a singleton
                    break
        elif sampling_method == "mesh":
            # create a mesh for the convex coefficients
            mesh = generate_mesh_over_simplex(resolution=resolution, x_dim=n_vertices, X=1)
            for coefficient_sample in mesh:
                y_sample = sum(
                    [prop_Y.vertex_flow[t - 1].vertices[i] * coefficient_sample[i] for i in range(n_vertices)])
                y_samples.append(y_sample)
                if t - 1 == 0:
                    # previous step is a singleton
                    break
        else:
            raise Exception("Sampling method not recognized!")

        for y_sample in y_samples:
            x_req = compute_x_req([y_sample])
            P_req = req_2_simplex(x_req=x_req, X=X)

            # check intersection of P_req and R_x
            _, _, intersection_flag = intersect(P_req, R_x_vertices)
            if not intersection_flag:
                y_g = deepcopy(y_sample)
                break

        t += 1

        return y_g
