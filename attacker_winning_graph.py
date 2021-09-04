import numpy as np
from blotto_prop import BlottoProp
from environments import Environment, generate_env_from_name
from find_C2C import compute_C2C_time
from copy import deepcopy
from convex_hull_algs import isInHull
from utils import generate_x_req_set


def check_attacker_winning_graph(env: Environment, sampling_method="random", n_samples=10, resolution=None):
    def random_sample(x_dim, X):
        x_sample = np.random.random(x_dim)
        x_sample *= X / sum(x_sample)
        assert abs(sum(x_sample) - X) < 1e-4
        return x_sample

    def generate_mesh(resolution, X, x_dim):
        mesh = [[]]
        for dim in range(x_dim-1):
            while len(mesh[0]) < dim + 1:
                node = mesh.pop(0)
                X_remain = X - sum(node)
                children = np.linspace(0, X_remain, int(X_remain / resolution) + 1)
                for child in children:
                    new_node = deepcopy(node)
                    new_node.append(child)
                    mesh.append(new_node)
        for point in mesh:
            point.append(X - sum(point))
        return mesh

    # def get_single_point_safe_vertices(X, y):
    #     y_dim = y.shape[0]
    #     Y = sum(y)
    #     vertices = []
    #     for i in range(y_dim):
    #         vertex = deepcopy(y)
    #         vertex[i] = X + vertex[i] - Y
    #         vertices.append(vertex)
    #     return vertices

    prop_X = BlottoProp(connectivity=env.connectivity_X, agent_name="Defender")
    prop_Y = BlottoProp(connectivity=env.connectivity_Y, agent_name="Attacker")

    x_dim = env.get_dimension()
    X = env.X
    y_dim = env.get_dimension()
    Y = env.Y

    if sampling_method == "random":
        y_sample_pairs = []
        for _ in range(n_samples):
            y0 = random_sample(y_dim, Y)
            y1 = random_sample(y_dim, Y)
            y_sample_pairs.append([y0, y1])

    elif sampling_method == "mesh":
        y_sample_pairs = []
        mesh0 = generate_mesh(resolution=resolution, X=Y, x_dim=y_dim)
        mesh1 = deepcopy(mesh0)
        for y0 in mesh0:
            for y1 in mesh1:
                y_sample_pairs.append([np.array(y0), np.array(y1)])

    # To test figure 4 scenario
    y_sample_pairs.append([np.array([0.9, 0.1, 0.0, 0.0, 0.0]), np.array([0.1, 0.9, 0.0, 0.0, 0.0])])

    for y_sample_pair in y_sample_pairs:
        y_s, y_g = y_sample_pair[0], y_sample_pair[1]

        # propagate y_s and y_g for one step
        prop_Y.set_initial_vertices(initial_vertices=[y_s])
        y_vertices_s = prop_Y.prop_step()
        prop_Y.set_initial_vertices(initial_vertices=[y_g])
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
        end_vertices = prop_X.prop_multi_steps(tau).vertices

        # check intersection
        # if safe_set intersect with terminal_hull, then at least one vertex of the safe set is in the terminal_hull
        intersect = False
        for safe_vertex in x_vertices_req_g.vertices:
            if isInHull(safe_vertex, deepcopy(end_vertices)):
                intersect = True
                break

        if not intersect:
            print("A solution is found!")
            print([y_s, y_g])
            return [y_s, y_g]

    print("No solution found!")
    return None


if __name__ == "__main__":
    env_name = "figure-4"
    env = generate_env_from_name(env_name)

    soln = check_attacker_winning_graph(env=env, n_samples=0, sampling_method="mesh", resolution=0.5)
