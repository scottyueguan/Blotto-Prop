import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from operator import itemgetter
from shapely.geometry import Polygon
from copy import deepcopy
from utils import Vertices, isEqual, isSamePoint
import itertools
from convex_hull_algs import remove_non_vertex_auxPoint, remove_non_vertex_analytic


class BlottoProp:
    def __init__(self, connectivity, agent_name, T=50, eps=0, hull_method="aux_point", need_connections=False):
        self.connectivity = connectivity
        self.N = len(self.connectivity)
        self.X = None
        self.T = T
        self.agent_name = agent_name

        self.vertex_flow = None
        self.extreme_actions = self.generate_extreme_actions()

        self.rotation_parameters = None
        self.eps = eps
        self.hull_method = hull_method
        self.need_connections = need_connections

    def set_initial_vertices(self, initial_vertices, perturb_singleton=True):
        self.X = sum(initial_vertices[0])
        self.vertex_flow = self._generate_initial_vertex(points=initial_vertices, perturb_singleton=perturb_singleton)

        self.rotation_parameters = self._init_coordinate_transferer()

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

    def __len__(self):
        return len(self.vertex_flow)

    def _generate_initial_vertex(self, points, perturb_singleton=True):
        if len(points) > 1:
            return [Vertices(points, None)]

        x0 = points[0]
        X = sum(x0)
        if not perturb_singleton:
            return [Vertices([x0], None)]
        else:
            perturbed_points = []
            for i in range(len(x0)):
                new_point = deepcopy(x0)
                new_point[i] += 1e-3
                new_point *= (X / sum(new_point))
                perturbed_points.append(new_point)
            return [Vertices(perturbed_points, None)]

    def append_flow(self, vertices: Vertices):
        self.vertex_flow.append(vertices)

    def override_flow(self, vertices: Vertices):
        self.vertex_flow[-1] = vertices

    def reset_flow(self):
        self.vertex_flow = None

    def revert_step(self):
        self.vertex_flow.pop()

    def prop_step(self):  # propagate one step
        new_vertices = []
        for x in self.vertex_flow[-1].vertices:
            new_vertices += self._prop_vertex(x)

        if len(self.vertex_flow[-1]) == 1:
            # for singleton no need to remove redundant vertices
            new_vertices = Vertices(vertices=new_vertices)
            new_vertices = self.remove_duplicated_points(new_vertices)
        else:
            if self.hull_method == "aux_point":
                new_vertices = remove_non_vertex_auxPoint(new_vertices, need_connections=self.need_connections)
            else:
                new_vertices = remove_non_vertex_analytic(new_vertices, rotation_parameters=self.rotation_parameters,
                                                          need_connections=self.need_connections)

        return new_vertices

    def remove_duplicated_points(self, points):
        new_points = [points[0]]
        for point in points:
            flag = False
            for existing_point in new_points:
                if isSamePoint(point, existing_point):
                    flag = True
                    break
            if not flag:
                new_points.append(point)

        return new_points

    def prop_multi_steps(self, t):
        for _ in range(t):
            new_vertices = self.prop_step()
            self.append_flow(new_vertices)
        return self.vertex_flow[-1]

    def cut(self, vertices, cut_vertices):

        current_vertices = vertices

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

        # self.vertex_flow[-1] = Vertices(new_points, new_connections)

        return Vertices(new_points, new_connections)

    def req_2_simplex(self, x_req):
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

    def plot_simplex(self, t, color='b'):

        r = self.X

        plt.figure(figsize=(6, 6), dpi=120)

        ax = plt.axes(projection='3d')
        ax.view_init(azim=50, elev=45)

        xline = np.linspace(0, r, 20)
        yline = r - xline
        zline = np.linspace(0, 0, 20)
        ax.plot3D(xline, yline, zline, color + '-', label="simplex")

        xline = np.linspace(0, 0, 20)
        yline = np.linspace(0, r, 20)
        zline = r - yline
        ax.plot3D(xline, yline, zline, 'b-')

        xline = np.linspace(0, r, 20)
        yline = np.linspace(0, 0, 20)
        zline = r - xline
        ax.plot3D(xline, yline, zline, 'b-')

        ax.set_xlim3d(0, r + 0.1)
        ax.set_ylim3d(0, r + 0.1)
        ax.set_zlim3d(0, r + 0.1)

        plt.title("Agent {} feasible region at time {}".format(self.agent_name, t), fontsize=20)

        return ax

    def _init_coordinate_transferer(self):
        o = np.zeros(self.N)
        o[0] = self.X

        A = np.zeros((self.N, self.N - 1))
        for i in range(1, self.N):
            A[i, i - 1] = 1
            A[0, i - 1] = -1

        A_pseudo_inv = np.linalg.pinv(A)

        rotation_parameters = {"o": o, "A_pseudo_inv": A_pseudo_inv, "A": A}

        return rotation_parameters
