import numpy as np
from blotto_prop import BlottoProp
from environments import Environment, generate_env_from_name
from find_C2C import compute_C2C_time
from copy import deepcopy
from convex_hull_algs import isInHull, intersect
from utils import generate_x_req_set
from tqdm import tqdm
from utils import generate_mesh_over_simplex, random_sample_over_simplex
from concurrent.futures import *


# TODO: Implement multi-thread
def check_attacker_winning_graph(env: Environment, sampling_method="random", n_samples=10, resolution=None):

    prop_X = BlottoProp(connectivity=env.connectivity_X, agent_name="Defender")
    prop_Y = BlottoProp(connectivity=env.connectivity_Y, agent_name="Attacker")

    x_dim = env.get_dimension_X()
    X = env.X
    y_dim = env.get_dimension_Y()
    Y = env.Y

    y_sample_pairs = []
    if sampling_method == "random":
        for _ in range(n_samples):
            y0 = random_sample_over_simplex(y_dim, Y)
            y1 = random_sample_over_simplex(y_dim, Y)
            y_sample_pairs.append([y0, y1])

    elif sampling_method == "mesh":
        mesh0 = generate_mesh_over_simplex(resolution=resolution, X=Y, x_dim=y_dim)
        mesh1 = deepcopy(mesh0)
        for y0 in mesh0:
            for y1 in mesh1:
                y_sample_pairs.append([np.array(y0), np.array(y1)])

    # To test figure 4 scenario
    # y_sample_pairs = [[np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])]]

    for i in tqdm(range(len(y_sample_pairs))):
        y_sample_pair = y_sample_pairs[i]
        y_s, y_g = y_sample_pair[0], y_sample_pair[1]

        # propagate y_s and y_g for one step
        prop_Y.set_initial_vertices(initial_vertices=[y_s], perturb_singleton=False)
        y_vertices_s = prop_Y.prop_step()
        prop_Y.set_initial_vertices(initial_vertices=[y_g], perturb_singleton=False)
        y_vertices_g = prop_Y.prop_step()

        # generate x req sets
        x_vertices_req_s = generate_x_req_set(y_vertices_s, X)
        x_vertices_req_g = generate_x_req_set(y_vertices_g, X)

        # get C2C time
        tau = compute_C2C_time(env, y_s, y_g, role="Y")

        # set x initial set
        prop_X.reset_flow()
        prop_X.set_initial_vertices(x_vertices_req_s.vertices, perturb_singleton=False)

        # propagate tau steps
        end_vertices = prop_X.prop_multi_steps(tau)

        # check intersection
        # if safe_set intersect with terminal_hull, then at least one vertex of the safe set is in the terminal_hull
        safe_vertices = x_vertices_req_g
        _, _, isIntersect = intersect(safe_vertices, end_vertices)

        if not isIntersect:
            print("A solution is found!")
            np.set_printoptions(precision=4)
            print("The attacker start point is {}".format(y_s))
            print("The attacker end point is {}".format(y_g))
            print("The attacker C2C time is {}".format(tau))
            return [y_s, y_g]

    print("No solution found!")
    return None


if __name__ == "__main__":
    env_name = "figure-4-v2"
    env = generate_env_from_name(env_name)
    env.set_X(X=4)
    env.set_Y(Y=1)

    soln = check_attacker_winning_graph(env=env, n_samples=0, sampling_method="mesh", resolution=0.3)
