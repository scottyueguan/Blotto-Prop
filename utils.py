import numpy as np
from operator import itemgetter
from copy import deepcopy
from typing import List


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

    def plot(self, ax, color, legend=None):
        vertices = self.vertices
        xdata = [vertices[i][0] for i in range(len(vertices))]
        ydata = [vertices[i][1] for i in range(len(vertices))]
        zdata = [vertices[i][2] for i in range(len(vertices))]


        ax.scatter3D(xdata, ydata, zdata, color=color, s=40)

        if self.connections is not None:
            for i, connection in enumerate(self.connections):
                start = connection[0]
                end = connection[1]
                xline = [vertices[start][0], vertices[end][0]]
                yline = [vertices[start][1], vertices[end][1]]
                zline = [vertices[start][2], vertices[end][2]]

                if i == 0:
                    ax.plot3D(xline, yline, zline, color + "-", label=legend)
                else:
                    ax.plot3D(xline, yline, zline, color + "-")

        ax.legend(loc='lower center', fontsize=18)

        return ax


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
    assert sum(x_req) <= X + 1e-5
    cut_points = []
    x_dim = x_req.shape[0]

    for i in range(x_dim):
        cut_vertex = deepcopy(x_req)
        cut_vertex[i] = X - sum(x_req) + x_req[i]  # cut_vertex = 1 - sum_{k!=i} x_req[i]
        cut_points.append(cut_vertex)

    connections = gen_standard_connection(x_dim)

    return Vertices(cut_points, connections)


def generate_x_req_set(vertices_y: Vertices, X):
    x_req = compute_x_req(vertices_y)
    x_req_vertices = req_2_simplex(x_req, X)
    return x_req_vertices


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
