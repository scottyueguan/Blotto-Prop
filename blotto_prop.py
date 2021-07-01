import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from operator import itemgetter
from shapely.geometry import Polygon
from copy import deepcopy
from utils import Vertices


class BlottoProp:
    def __init__(self, connectivity, x0, T, agent_name):
        self.connectivity = connectivity
        self.N = len(self.connectivity)
        self.x0 = x0
        self.T = T
        self.agent_name = agent_name

        self.vertex_flow = [Vertices([self.x0], None)]
        self.extreme_actions = self.generate_extreme_actions()

        self.o, self.A, self.A_pseudo_inv = self._init_coordinate_transferer()

    # def prop_T(self): # propogate the whole T steps
    #     self.plot_feasible_region(self.vertex_flow[0], 0)
    #
    #     for t in range(self.T - 1):
    #         new_vertices = []
    #         for x in self.vertex_flow[t-1]:
    #             new_vertices += self._prop_vertex(x)
    #
    #         new_vertices = self._remove_non_vertex(new_vertices)
    #         self.vertex_flow.append(new_vertices)
    #
    #         self.plot_feasible_region(new_vertices, t + 1)

    def prop_step(self):  # propogate one step
        new_vertices = []
        for x in self.vertex_flow[-1].vertices:
            new_vertices += self._prop_vertex(x)

        new_vertices, connection = self._remove_non_vertex(new_vertices)
        self.vertex_flow.append(Vertices(new_vertices, connection))
        return new_vertices

    def cut(self, cut_vertices):

        current_vertices = self.vertex_flow[-1]

        cut_vertices_rotate = self._rotate_points(cut_vertices.vertices)
        current_vertices_rotate = self._rotate_points(current_vertices.vertices)

        p1 = Polygon(cut_vertices_rotate)
        p2 = Polygon(current_vertices_rotate)

        p_new = p1.intersection(p2)

        # plot projected geometries
        # plt.plot(*p1.exterior.xy)
        # plt.plot(*p2.exterior.xy)
        # plt.plot(*p_new.exterior.xy)
        #
        # plt.show()

        new_points_tmp = p_new.exterior.coords.xy
        new_points_rotated = [np.array([new_points_tmp[0][i], new_points_tmp[1][i]]) for i in
                                range(len(new_points_tmp[0]) - 1)]
        new_points = self._rotate_back_points(new_points_rotated)
        new_connections = self._gen_standard_connection(len(new_points))

        self.vertex_flow[-1] = Vertices(new_points, new_connections)

        return Vertices(new_points, new_connections)

    def x_req_2_simplex(self, x_req):
        assert sum(x_req) < 1
        cut_points = []

        for i in range(self.N):
            cut_vertex = deepcopy(x_req)
            cut_vertex[i] = 1 - sum(x_req) + x_req[i]  # cut_vertex = 1 - sum_{k!=i} x_req[i]
            cut_points.append(cut_vertex)

        connections = self._gen_standard_connection(self.N)

        return Vertices(cut_points, connections)

    def _gen_standard_connection(self, n):
        connections = []
        for i in range(n):
            if i + 1 <= n - 1:
                connections.append([i, i + 1])
            else:
                connections.append([i, 0])
        return connections

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
        connections = [
            np.concatenate((np.where(vertex_index == simplex[0])[0], np.where(vertex_index == simplex[1])[0])) for
            simplex in hull.simplices]
        return new_vertices, connections

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

    def plot_simplex(self, t):

        plt.figure(figsize=(8,6), dpi=80)

        ax = plt.axes(projection='3d')
        ax.view_init(azim=50, elev=45)

        xline = np.linspace(0, 1, 20)
        yline = 1 - xline
        zline = np.linspace(0, 0, 20)
        ax.plot3D(xline, yline, zline, 'b-', label="simplex")

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

        plt.title("Agent {} feasible region at time {}".format(self.agent_name, t))

        return ax

    def _init_coordinate_transferer(self):
        o = np.zeros(self.N)
        o[0] = 1

        A = np.zeros((self.N, self.N - 1))
        for i in range(1, self.N):
            A[i, i - 1] = 1
            A[0, i - 1] = -1

        A_pseudo_inv = np.linalg.pinv(A)

        return o, A, A_pseudo_inv
