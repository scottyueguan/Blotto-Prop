import numpy as np
from environments import generate_env_from_name
from blotto_prop import BlottoProp

env_name = "figure-4"
env = generate_env_from_name(env_name)

prop = BlottoProp(connectivity=env.connectivity_X, T=env.Tf, agent_name="Defender",
                  hull_method="aux_point", need_connections=False)

extreme_actions = prop.extreme_actions

for _ in range(20):
    n_extreme_actions = len(extreme_actions)
    random_coeff = np.random.random(n_extreme_actions)
    random_coeff /= sum(random_coeff)
    new_action = np.sum(extreme_actions[i] * random_coeff [i] for i in range(n_extreme_actions))

    computed_coeff = np.zeros(n_extreme_actions)

    for k in range(n_extreme_actions):
        extreme_action = extreme_actions[k]
        coeff = 1
        for i in range(env.get_dimension_X()):
            for j in range(env.get_dimension_Y()):
                if extreme_action[i, j] == 1:
                    coeff *= new_action[i,j]
        computed_coeff[k] = coeff

    recovered_action = np.sum(extreme_actions[i] * computed_coeff[i] for i in range(n_extreme_actions))

    print(sum(sum(abs(new_action - recovered_action))))
