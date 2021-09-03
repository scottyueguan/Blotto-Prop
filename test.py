from scipy.spatial import ConvexHull

points = [[0, 1], [1, 0], [0.5, 0.5], [1, 1]]

hull = ConvexHull(points)

print('done')
