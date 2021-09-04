import numpy as np
from scipy.spatial import ConvexHull
from operator import itemgetter
import itertools


def convex_hull(points, aux_indices=None, need_connections=False):
    def generate_connections(hull, aux_indices=None):
        def generate_connections_from_simplex(simplex):
            connections = list(itertools.combinations(simplex, 2))
            return connections

        def contain_aux_points(simplex, aux_indices):
            if aux_indices is None:
                return False
            aux_indices = set(aux_indices)
            simplex = set(simplex)
            if simplex.intersection(aux_indices):
                return True
            else:
                return False

        vertrex_index = hull.vertices
        n_points = hull.npoints
        mapping = []
        used_index = 0
        for index in range(n_points):
            if index in vertrex_index:
                mapping.append(used_index)
                used_index += 1
            else:
                mapping.append(None)

        simplices = hull.simplices
        connections = []
        non_boundary = []
        for simplex in simplices:
            if not contain_aux_points(simplex, aux_indices):
                simplex_connections = generate_connections_from_simplex(simplex)
                for simplex_connection in simplex_connections:
                    new_connection = (mapping[simplex_connection[0]], mapping[simplex_connection[1]])
                    if new_connection[0] > new_connection[1]:
                        new_connection = (new_connection[1], new_connection[0])
                    if new_connection in connections:
                        non_boundary.append(connections.index(new_connection))
                    else:
                        connections.append(new_connection)
        for non_index in sorted(non_boundary, reverse=True):
            connections.pop(non_index)

        return list(connections)

    hull = ConvexHull(points)
    vertex_index = set(hull.vertices) - set(aux_indices)
    new_vertices = list(itemgetter(*vertex_index)(points))

    if need_connections:
        connections = generate_connections(hull, aux_indices=aux_indices)
    else:
        connections = None
    return new_vertices, connections


def remove_non_vertex_auxPoint(vertices, need_connections):
    def add_aux_point(points):
        example = vertices[0]
        aux_point = np.zeros(example.shape) + sum(example)
        points.append(aux_point)
        aux_points = [aux_point]
        aux_indices = [len(points) - 1]
        return points, aux_points, aux_indices

    vertices_addApoint, aux_points, aux_indices = add_aux_point(vertices)
    new_vertices, new_connections = convex_hull(vertices_addApoint, need_connections=need_connections,
                                                aux_indices=aux_indices)
    return new_vertices, new_connections


def remove_non_vertex_analytic(vertices, need_connections, rotation_parameters):
    def rotate_points(points):
        o = rotation_parameters["o"]
        A_pseudo_inv = rotation_parameters["A_pseudo_inv"]

        rotated_points = []
        for point in points:
            diff = point - o
            rotated_point = np.matmul(A_pseudo_inv, diff.T)
            rotated_points.append(rotated_point.T)

        return rotated_points

    def rotate_back_points(rotated_points):
        o = rotation_parameters["o"]
        A = rotation_parameters["A"]

        original_points = []
        for rotated_point in rotated_points:
            diff = np.matmul(A, rotated_point.T).T
            point = o + diff
            original_points.append(point)
        return original_points

    rotated_vertices = rotate_points(vertices)
    new_vertices, connections = convex_hull(rotated_vertices, need_connections=need_connections,
                                            aux_indices=None)
    final_vertices = rotate_back_points(new_vertices)
    return final_vertices, connections


def isInHull(point, vertices):
    def isInVertices(point, vertices):
        for vertex in vertices:
            if np.linalg.norm(point - vertex) < 1e-4:
                return True
        return False

    if isInVertices(point, vertices):
        return True

    vertices.append(point)

    hull_vertices, _ = remove_non_vertex_auxPoint(vertices, False)

    if isInVertices(point, hull_vertices):
        return False
    else:
        return True
