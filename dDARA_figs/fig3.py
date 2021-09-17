from environments import generate_env_from_name
from blotto_prop import BlottoProp
import numpy as np
import matplotlib.pyplot as plt
from utils import generate_x_req_set
from convex_hull_algs import intersect
from copy import deepcopy
import os
from utils import FIG_PATH

x0 = np.array([7, 2, 1])
y0 = np.array([2, 1, 1])

env = generate_env_from_name('fig-2')

prop_X = BlottoProp(connectivity=env.connectivity_X, agent_name='Defender', T=2, hull_method='aux_point',
                    need_connections=True)
prop_X.set_initial_vertices(initial_vertices=[x0], perturb_singleton=False)

prop_Y = BlottoProp(connectivity=env.connectivity_Y, agent_name='Attacker', T=2, hull_method='aux_point',
                    need_connections=True)
prop_Y.set_initial_vertices(initial_vertices=[y0], perturb_singleton=False)

x_vertices_1 = prop_X.prop_step()
y_vertices_1 = prop_Y.prop_step()

x_vertices_req_1 = generate_x_req_set(vertices_y=y_vertices_1, X=env.X)
x_safe_vertices_1, _, found = intersect(vertices1=x_vertices_req_1, vertices2=deepcopy(x_vertices_1), sloppy=False,
                                        need_connections=True)

# plot reachable sets
fig_X, ax_X = prop_X.plot_simplex(t=1, color='b', axis_limit=env.X + 0.1)
x_vertices_0 = prop_X.vertex_flow[0]
ax_X = x_vertices_0.plot(ax=ax_X, color='b', legend='initial state')
ax_X = x_vertices_1.plot(ax=ax_X, color='c', legend='reachable set', shade=False)
ax_X = x_safe_vertices_1.plot(ax=ax_X, color='g', line_style='-', legend='safe set', shade=True, alpha=0.5)
ax_X = x_vertices_req_1.plot(ax=ax_X, color='y', line_style='--', legend='required resource', shade=False,
                             plot_vertices=False)

fig_Y, ax_Y = prop_Y.plot_simplex(t=1, color='r', axis_limit=env.X + 0.1)
y_vertices_0 = prop_Y.vertex_flow[0]
ax_Y = y_vertices_0.plot(ax=ax_Y, color='r', legend='initial state')
ax_Y = y_vertices_1.plot(ax=ax_Y, color='m', legend='reachable set', shade=True)

plt.show()


file_name_X = 'figure-3b.pdf'
fig_X.savefig(os.path.join(FIG_PATH, file_name_X), dpi=120)
file_name_Y = 'figure-3a.pdf'
fig_Y.savefig(os.path.join(FIG_PATH, file_name_Y), dpi=120)