import numpy as np
from blotto_prop import BlottoProp
from convex_hull_algs import isInHull
from copy import deepcopy
from environments import Environment, generate_env_from_name
from utils.utils import isEqual


def compute_C2C_time(env: Environment, x_s, x_g, role):
    if role == "X":
        prop = BlottoProp(connectivity=env.connectivity_X, T=env.Tf, agent_name="Defender",
                          hull_method="aux_point",
                          need_connections=False)
        total_resource = env.X
    elif role == "Y":
        prop = BlottoProp(connectivity=env.connectivity_Y, T=env.Tf, agent_name="Attacker",
                          hull_method="aux_point",
                          need_connections=False)
        total_resource = env.Y
    else:
        raise Exception("Role str error!")

    assert isEqual(sum(x_s), total_resource) and isEqual(sum(x_g), total_resource)

    assert isEqual(sum(x_s), total_resource) and isEqual(sum(x_g), total_resource)

    prop.set_initial_vertices(initial_vertices=[x_s], perturb_singleton=True)

    tau = 0

    for t in range(env.Tf + 1):
        tau += 1
        new_vertices = prop.prop_step()
        if isInHull(x_g, deepcopy(new_vertices.vertices)):
            return tau
        prop.append_flow(new_vertices)

    return None


if __name__ == "__main__":
    env_name = "figure-4-v2"
    env = generate_env_from_name(env_name)

    x_s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0])
    x_g = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])

    tau = compute_C2C_time(env=env, x_s=x_s, x_g=x_g, role="X")

    # x_s = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    # x_g = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    #
    # tau = compute_C2C_time(env_name=env_name, x_s=x_s, x_g=x_g, role="Y")

    print("C2C time is {}".format(tau))
