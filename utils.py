import numpy as np
from operator import itemgetter


class Vertices:
    def __init__(self, vertices, connections):
        self.vertices = vertices
        self.connections = connections

    def __len__(self):
        return len(self.vertices)

    def plot(self, ax, color, legend=None):
        vertices = self.vertices
        xdata = [vertices[i][0] for i in range(len(vertices))]
        ydata = [vertices[i][1] for i in range(len(vertices))]
        zdata = [vertices[i][2] for i in range(len(vertices))]

        ax.scatter3D(xdata, ydata, zdata, color=color, label='{} vertices'.format(legend), s=40)

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


def compute_x_req(vertices_y: Vertices):
    N = len(vertices_y.vertices[0])
    x_req = np.zeros(N)
    for i in range(N):
        vertices_y_i = [vertices_y.vertices[k][i] for k in range(len(vertices_y))]
        x_req[i] = max(vertices_y_i)
    return x_req


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
