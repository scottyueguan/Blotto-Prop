import numpy as np
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull
import scipy as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
from sympy import Plane, Point3D
import networkx as nx


def simplify(triangles):
    """
    Simplify an iterable of triangles such that adjacent and coplanar triangles form a single face.
    Each triangle is a set of 3 points in 3D space.
    """

    # create a graph in which nodes represent triangles;
    # nodes are connected if the corresponding triangles are adjacent and coplanar
    G = nx.Graph()
    G.add_nodes_from(range(len(triangles)))
    for ii, a in enumerate(triangles):
        for jj, b in enumerate(triangles):
            if (ii < jj):  # test relationships only in one way as adjacency and co-planarity are bijective
                if is_adjacent(a, b):
                    if is_coplanar(a, b, np.pi / 90.):
                        G.add_edge(ii, jj)

    # triangles that belong to a connected component can be combined
    components = list(nx.connected_components(G))
    simplified = [set(flatten(triangles[index] for index in component)) for component in components]

    # need to reorder nodes so that patches are plotted correctly
    reordered = [reorder(face) for face in simplified]

    return reordered


def is_adjacent(a, b):
    return len(set(a) & set(b)) == 2  # i.e. triangles share 2 points and hence a side


def is_coplanar(a, b, tolerance_in_radians=0):
    a1, a2, a3 = a
    b1, b2, b3 = b
    plane_a = Plane(Point3D(a1), Point3D(a2), Point3D(a3))
    plane_b = Plane(Point3D(b1), Point3D(b2), Point3D(b3))
    if not tolerance_in_radians:  # only accept exact results
        return plane_a.is_coplanar(plane_b)
    else:
        angle = plane_a.angle_between(plane_b).evalf()
        angle %= np.pi  # make sure that angle is between 0 and np.pi
        return (angle - tolerance_in_radians <= 0.) or \
               ((np.pi - angle) - tolerance_in_radians <= 0.)


flatten = lambda l: [item for sublist in l for item in sublist]


def reorder(vertices):
    """
    Reorder nodes such that the resulting path corresponds to the "hull" of the set of points.

    Note:
    -----
    Not tested on edge cases, and likely to break.
    Probably only works for convex shapes.

    """
    if len(vertices) <= 3:  # just a triangle
        return vertices
    else:
        # take random vertex (here simply the first)
        reordered = [vertices.pop()]
        # get next closest vertex that is not yet reordered
        # repeat until only one vertex remains in original list
        vertices = list(vertices)
        while len(vertices) > 1:
            idx = np.argmin(get_distance(reordered[-1], vertices))
            v = vertices.pop(idx)
            reordered.append(v)
        # add remaining vertex to output
        reordered += vertices
        return reordered


def get_distance(v1, v2):
    v2 = np.array(list(v2))
    difference = v2 - v1
    ssd = np.sum(difference ** 2, axis=1)
    return np.sqrt(ssd)


def plot_polyhedron(vertices, ax, color, ax_lim=8):
    verts = np.array(vertices)
    verts = np.around(verts, decimals=5)
    hull = ConvexHull(verts)
    faces = hull.simplices

    ax.dist = 10
    ax.azim = 30
    ax.elev = 10
    ax.set_xlim([0, ax_lim])
    ax.set_ylim([0, ax_lim])
    ax.set_zlim([0, ax_lim])

    triangles = []
    for s in faces:
        sq = [
            (verts[s[0], 0], verts[s[0], 1], verts[s[0], 2]),
            (verts[s[1], 0], verts[s[1], 1], verts[s[1], 2]),
            (verts[s[2], 0], verts[s[2], 1], verts[s[2], 2])
        ]
        triangles.append(sq)

    new_faces_ = simplify(triangles)
    new_faces = [list(new_face) for new_face in new_faces_]
    for sq in new_faces:
        f = a3.art3d.Poly3DCollection([sq])
        f.set_color(color)
        f.set_edgecolor('k')
        f.set_alpha(0.1)
        ax.add_collection3d(f)

    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], marker='o', color=color)

    return new_faces

if __name__ == "__main__":
    import numpy as np
    from numpy import array
    vertices = [array([2., 1., 0.]), array([1., 1., 1.]), array([0., 2., 1.]), array([0., 3., 0.]), array([1., 1., 6.]), array([0., 2., 6.]), array([0., 8., 0.]), array([7., 1., 0.])]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_polyhedron(vertices=vertices, ax=ax, color='b')
    plt.show()
