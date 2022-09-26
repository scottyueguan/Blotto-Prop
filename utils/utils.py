import numpy as np
from operator import itemgetter
from copy import deepcopy
from typing import List
import mpl_toolkits.mplot3d as mp3d
from matplotlib.colors import to_rgb as c2rgb
import os

ROOT_PATH = os.path.dirname(__file__)
FIG_PATH = os.path.join(ROOT_PATH, 'figures')


class Vertices:
    def __init__(self, vertices: List, connections=None, equations=None):
        self.vertices = vertices
        self.connections = connections
        self.equations = equations

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        self.ix = 0
        return iter(self.vertices)

    # def __next__(self):
    #     if self.ix == len(self.vertices):
    #         raise StopIteration
    #     else:
    #         item = self.vertices[self.ix]
    #         self.ix += 1
    #         return item

    def __getitem__(self, item):
        return self.vertices[item]

    def append(self, item):
        self.vertices.append(item)

    @property
    def has_equations(self):
        if self.equations is not None:
            return True
        else:
            return False

    def plot(self, ax, color, line_style='-', line_width=None, legend=None, shade=False, alpha=0.2, plot_vertices=True):
        vertices = self.vertices
        xdata = [vertices[i][0] for i in range(len(vertices))]
        ydata = [vertices[i][1] for i in range(len(vertices))]
        zdata = [vertices[i][2] for i in range(len(vertices))]

        if not plot_vertices:
            pass
        elif self.connections is None:
            ax.scatter3D(xdata, ydata, zdata, color=color, s=40, label=legend)
        else:
            ax.scatter3D(xdata, ydata, zdata, color=color, s=40)

        if shade:
            points = self._reorder_vertices()
            face = mp3d.art3d.Poly3DCollection([points], alpha=alpha, linewidth=0)
            rgb = c2rgb(color)
            face.set_facecolor((*rgb, alpha))
            ax.add_collection3d(face)

        if self.connections is not None:
            for i, connection in enumerate(self.connections):
                start = connection[0]
                end = connection[1]
                xline = [vertices[start][0], vertices[end][0]]
                yline = [vertices[start][1], vertices[end][1]]
                zline = [vertices[start][2], vertices[end][2]]

                if i == 0:
                    ax.plot3D(xline, yline, zline, color + line_style, label=legend, linewidth=line_width)
                else:
                    ax.plot3D(xline, yline, zline, color + line_style, linewidth=line_width)

        ax.legend(loc='upper left', fontsize=15)

        return ax

    def _reorder_vertices(self):
        new_vertices = self.vertices
        if self.connections is not None:
            neighbors = [[] for _ in range(len(self.vertices))]
            for connection in self.connections:
                neighbors[connection[0]].append(connection[1])
                neighbors[connection[1]].append(connection[0])

            loop = [-100, 0]
            for i in range(1, len(self.vertices) + 1):
                for neighbor in neighbors[loop[i]]:
                    if neighbor != loop[i - 1]:
                        loop.append(neighbor)
                        break

            loop.pop(0)
            loop.pop(-1)

            new_vertices = [tuple(self.vertices[loop[i]]) for i in range(len(loop))]
        return new_vertices


def compare_vertices(vertices1: Vertices, vertices2: Vertices):
    points1 = vertices1.vertices
    points2 = vertices2.vertices

    hash1 = sum([np.linalg.norm(points1[k]) for k in range(len(points1))])
    hash2 = sum([np.linalg.norm(points2[k]) for k in range(len(points2))])

    if isEqual(hash1, hash2):
        return True
    else:
        return False


def isSamePoint(point1, point2):
    diff = point1 - point2
    norm = np.linalg.norm(diff)
    if isEqual(norm, 0):
        return True
    else:
        return False


def gen_standard_connection(n):
    connections = []
    for i in range(n):
        if i + 1 <= n - 1:
            connections.append([i, i + 1])
        else:
            connections.append([i, 0])
    return connections


def isInteger(point):
    integer_flag = True
    for x in point:
        if not (isEqual(A=x % 1, B=0) or isEqual(A=x % 1, B=1)):
            integer_flag = False
            break
    return integer_flag


def compute_x_req(vertices_y):
    if isinstance(vertices_y, Vertices):
        vertices_y = vertices_y.vertices

    N = len(vertices_y[0])
    x_req = np.zeros(N)
    for i in range(N):
        vertices_y_i = [vertices_y[k][i] for k in range(len(vertices_y))]
        x_req[i] = max(vertices_y_i)
    return x_req


def req_2_simplex(x_req, X):
    from convex_hull_algs import convex_hull
    assert sum(x_req) <= X + 1e-5
    cut_points = []
    x_dim = x_req.shape[0]

    for i in range(x_dim):
        cut_vertex = deepcopy(x_req)
        cut_vertex[i] = X - sum(x_req) + x_req[i]  # cut_vertex = 1 - sum_{k!=i} x_req[i]
        cut_points.append(cut_vertex)

    polytope, success = convex_hull(points=cut_points, need_connections=True, need_equations=True)
    assert success

    return polytope


def req_cut(x_req, max_surplus=30):
    from convex_hull_algs import convex_hull
    N = x_req.shape[0]

    vertices = [x_req]
    for i in range(N):
        cut_vertex = deepcopy(x_req)
        cut_vertex[i] = 0
        cut_vertex[i] = N - np.sum(cut_vertex) + max_surplus
        vertices.append(cut_vertex)

    polytope, success = convex_hull(points=vertices, need_connections=True, need_equations=True)
    assert success

    return polytope


def generate_x_req_set(vertices_y: Vertices, X):
    degenerate = False
    x_req = compute_x_req(vertices_y)
    if np.sum(x_req) == len(vertices_y.vertices[0]):
        degenerate = True
        return None, degenerate
    else:
        if X is not None:
            x_req_vertices = req_2_simplex(x_req, X)
        else:
            x_req_vertices = req_cut(x_req)
        return x_req_vertices, degenerate


def compute_y_req_v1(vertices_x: Vertices, eta=0.2):
    N = len(vertices_x.vertices[0])
    y_req = np.zeros(N)
    for i in range(N):
        vertices_x_i = [vertices_x.vertices[k][i] for k in range(len(vertices_x))]
        y_req[i] = eta * max(vertices_x_i)
    return y_req


def compute_y_req_v2(vertices_x: Vertices, eta=0.1):
    N = len(vertices_x.vertices[0])
    assert eta * N < 1  # check if eta is too large

    y_req = np.zeros(N)
    for i in range(N):
        vertices_x_i = [vertices_x.vertices[k][i] for k in range(len(vertices_x))]
        if max(vertices_x_i) > 0:
            y_req[i] = eta
        else:
            y_req[i] = 0
    return y_req


def isEqual(A, B, eps=1e-6):
    if abs(A - B) < eps:
        return True
    else:
        return False


def isSingleton_eps(items, eps=1e-4):
    if len(items) < 0:
        raise Exception("Empty list")

    example = list(items)[0]
    for item in items:
        if not isEqual(item, example):
            return False
    return True


def generate_mesh_over_simplex(resolution, X, x_dim):
    mesh = [[]]
    for dim in range(x_dim - 1):
        while len(mesh[0]) < dim + 1:
            node = mesh.pop(0)
            X_remain = X - sum(node)
            children = np.linspace(0, X_remain, int(X_remain / resolution) + 1)
            for child in children:
                new_node = deepcopy(node)
                new_node.append(child)
                mesh.append(new_node)
    for point in mesh:
        point.append(X - sum(point))
    return mesh


def random_sample_over_simplex(x_dim, X):
    x_sample = np.random.random(x_dim)
    x_sample *= X / sum(x_sample)
    assert abs(sum(x_sample) - X) < 1e-4
    return x_sample


def gen_standard_connection(n):
    connections = []
    for i in range(n):
        if i + 1 <= n - 1:
            connections.append([i, i + 1])
        else:
            connections.append([i, 0])
    return connections


def remove_duplicated_points(points):
    new_points = np.unique(points, axis=0)
    return new_points


if __name__ == "__main__":
    points = [np.array([1, 8, 3, 3, 4]), np.array([1, 8, 3, 3, 4]), np.array([1, 2, 3, 3, 4])]
    new_points = remove_duplicated_points(points)
    print(new_points)
