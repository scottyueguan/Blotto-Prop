import matplotlib.pyplot as plt
import numpy as np
from convex_hull_algs import remove_non_vertex_auxPoint, intersect

points_1 = [np.array([0.8, 0.1, 0.1, 0.0]), np.array([0.1, 0.8, 0.0, 0.1]), np.array([0.1, 0.0, 0.1, 0.8]),
            np.array([0.0, 0.1, 0.8, 0.1])]
points_2 = [np.array([0.3, 0.3, 0.4, 0.0]), np.array([0.3, 0.3, 0.0, 0.4]), np.array([0.3, 0.0, 0.3, 0.4]),
            np.array([0.0, 0.3, 0.3, 0.4])]

vertices_1, success_1 = remove_non_vertex_auxPoint(points_1, need_connections=True)
vertices_2, success_2 = remove_non_vertex_auxPoint(points_2, need_connections=True)
vertices, rays, found = intersect(vertices_1, vertices_2)

vertices_new, success = remove_non_vertex_auxPoint(vertices.vertices, need_connections=True)

# Plot
plt.figure(figsize=(6, 6), dpi=120)
ax = plt.axes(projection='3d')
ax.view_init(azim=50, elev=45)

vertices_1.plot(ax=ax, color='b')
vertices_2.plot(ax=ax, color='y')
vertices_new.plot(ax=ax, color='r')

plt.show()
