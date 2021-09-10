from utils import Vertices
import matplotlib.pyplot as plt
import numpy as np
from convex_hull_algs import remove_non_vertex_auxPoint

points_3 = []
for _ in range(10):
    point = np.random.random(4)
    point /= sum(point)
    points_3.append(point)

points_3 += [np.array([0.8, 0.1, 0.1, 0.0]), np.array([0.1, 0.8, 0.0, 0.1]), np.array([0.1, 0.0, 0.1, 0.8]),
             np.array([0.0, 0.1, 0.8, 0.1])]

vertices_3 = Vertices(vertices=points_3)
plt.figure(figsize=(6, 6), dpi=120)
ax = plt.axes(projection='3d')
ax.view_init(azim=50, elev=45)
vertices_3.plot(ax=ax, color='b', legend='Points')

hull_3, success = remove_non_vertex_auxPoint(points_3, need_connections=False)
hull_3.plot(ax=ax, color='r', legend='Hull Vertices')
plt.show()