import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from operator import itemgetter


class BlottoProp:
    def __init__(self, connectivity, x0, T):
        self.connectivity = connectivity
        self.N = len(self.connectivity)
        self.x0 = x0
        self.T = T
        self.vertex_flow = []
        self.extreme_actions = self.generate_extreme_actions()

        self.o, self.A, self.A_pseudo_inv = self._init_coordinate_transferer()

    def prop(self):
        vertices = [self.x0]
        self.plot_feasible_region(vertices, 0)

        for t in range(self.T - 1):
            new_vertices = []
            for x in vertices:
                new_vertices += self._prop_vertex(x)

            new_vertices = self._remove_non_vertex(new_vertices)
            self.vertex_flow.append(new_vertices)
            vertices = new_vertices

            self.plot_feasible_region(new_vertices, t + 1)

    def _prop_vertex(self, x):
        new_vertices = []
        for extreme_actions in self.extreme_actions:
            new_vertices.append(np.matmul(x, extreme_actions))
        return new_vertices

    def _remove_non_vertex(self, vertices):
        rotated_vertices = self._rotate_points(vertices)
        hull = ConvexHull(rotated_vertices)
        vertex_index = hull.vertices
        new_vertices = list(itemgetter(*vertex_index)(vertices))
        return new_vertices

    def _rotate_points(self, points):
        o = self.o
        A_pseudo_inv = self.A_pseudo_inv

        rotated_points = []
        for point in points:
            diff = point - o
            rotated_point = np.matmul(A_pseudo_inv, diff.T)
            rotated_points.append(rotated_point.T)

        return rotated_points

    def _rotate_back_points(self, rotated_points):
        o = self.o
        A = self.A

        original_points = []
        for rotated_point in rotated_points:
            diff = np.matmul(A, rotated_point.T).T
            point = o + diff
            original_points.append(point)
        return original_points

    def generate_extreme_actions(self):
        return self._expand(0, list=[np.array([])])

    def _expand(self, n, list):
        n_children = sum(self.connectivity[n, :])
        non_zero_indices = np.nonzero(self.connectivity[n, :])[0]

        for i in range(len(list)):
            current_list = list.pop(0)

            for j in range(n_children):
                new_row = np.zeros(self.N)
                new_row[non_zero_indices[j]] = 1
                if n > 0:
                    new_action = np.vstack((current_list, new_row))
                else:
                    new_action = new_row
                list.append(new_action)

        if n == self.N - 1:
            return list
        else:
            list = self._expand(n + 1, list)

        return list

    def plot_feasible_region(self, new_vertices, t):
        xdata = [new_vertices[i][0] for i in range(len(new_vertices))]
        ydata = [new_vertices[i][1] for i in range(len(new_vertices))]
        zdata = [new_vertices[i][2] for i in range(len(new_vertices))]

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xdata, ydata, zdata, color='r')
        ax.view_init(azim=50, elev=45)

        xline = np.linspace(0, 1, 20)
        yline = 1 - xline
        zline = np.linspace(0, 0, 20)
        ax.plot3D(xline, yline, zline, 'b-')

        xline = np.linspace(0, 0, 20)
        yline = np.linspace(0, 1, 20)
        zline = 1 - yline
        ax.plot3D(xline, yline, zline, 'b-')

        xline = np.linspace(0, 1, 20)
        yline = np.linspace(0, 0, 20)
        zline = 1 - xline
        ax.plot3D(xline, yline, zline, 'b-')

        ax.set_xlim3d(0, 1.1)
        ax.set_ylim3d(0, 1.1)
        ax.set_zlim3d(0, 1.1)

        plt.title("Feasible region at time {}".format(t))

        plt.show()

    def _init_coordinate_transferer(self):
        o = np.zeros(self.N)
        o[0] = 1

        A = np.zeros((self.N, self.N - 1))
        for i in range(1, self.N):
            A[i, i - 1] = 1
            A[0, i - 1] = -1

        A_pseudo_inv = np.linalg.pinv(A)

        return o, A, A_pseudo_inv
