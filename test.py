from convex_hull_algs import isInHull
import numpy as np

points = [np.array([0, 1]), np.array([1, 0]), np.array([0.5, 0.5]), np.array([1, 1])]

point = np.array([0.3, 0.7])

result = isInHull(point, points)

print(result)
