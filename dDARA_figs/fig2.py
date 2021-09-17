from environments import generate_env_from_name
from blotto_prop import BlottoProp
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import FIG_PATH

x0 = np.array([7, 2, 1])

env = generate_env_from_name('fig-2')

prop_X = BlottoProp(connectivity=env.connectivity_X, agent_name='Defender', T=2, hull_method='aux_point',
                    need_connections=True)

prop_X.set_initial_vertices(initial_vertices=[x0], perturb_singleton=False)

x_vertices_1 = prop_X.prop_step()

# plot reachable sets
fig, ax = prop_X.plot_simplex(t=1)
x_vertices_0 = prop_X.vertex_flow[0]
ax = x_vertices_0.plot(ax=ax, color='b', legend='initial state')
ax = x_vertices_1.plot(ax=ax, color='c', legend='reachable set', shade=True)

plt.show()

file_name = 'figure-2.pdf'
fig.savefig(os.path.join(FIG_PATH, file_name), dpi=120)
