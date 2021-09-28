from convex_hull_algs import convex_hull, remove_non_vertex_auxPoint, con2vert, intersect
import numpy as np
from utils import Vertices, gen_standard_connection
import matplotlib.pyplot as plt

# points = [np.array([0.0, 1.0, 0]), np.array([0, 0.8, 0.2]), np.array([0, 0, 1])]
#
# vertices = remove_non_vertex_auxPoint(points, need_connections=False, need_equations=True)
#
# vertices, rays, found = con2vert(A=vertices.equations["A"], b=vertices.equations["b"])
#
# vertices1 = Vertices(vertices=[np.array([1,3]), np.array([3,1])])
# vertices2 = Vertices(vertices=[np.array([2,2]), np.array([4,0])])
#
#
#
# vertices, rays, found = intersect(vertices1, vertices2)
#
# print(vertices, found)


# import cdd
# # mat = cdd.Matrix([[2,-1,-1,0],[0,1,0,0],[0,0,1,0]], number_type='fraction')
# mat = cdd.Matrix([[1,-1,0],[-1,1,0],[2,0,-1],[-1,0,1]], number_type='fraction')
# mat.rep_type = cdd.RepType.INEQUALITY
# poly = cdd.Polyhedron(mat)
#
# ext = poly.get_generators()
# print(ext)
print(np.random.random())