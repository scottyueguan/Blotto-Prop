import numpy as np
from graph_generators import generate_graph
from matplotlib import pyplot as plt
from q_prop import QProp
from scipy import io
from utils.plot_polyhedron import plot_polyhedron

# def plot_polyhedron(verticies: List[np.ndarray], ax, color):
#     polytope = np.array(verticies)
#
#     hull = ConvexHull(polytope)
#     for s in hull.simplices:
#         tri = Poly3DCollection(polytope[s])
#         tri.set_color(color)
#         tri.set_alpha(0.1)
#         tri.set_edgecolor('none')
#         ax.add_collection3d(tri)
#         edges = []
#         if distance.euclidean(polytope[s[0]], polytope[s[1]]) < distance.euclidean(polytope[s[1]],
#                                                                                    polytope[s[2]]):
#             edges.append((s[0], s[1]))
#             if distance.euclidean(polytope[s[1]], polytope[s[2]]) < distance.euclidean(polytope[s[2]],
#                                                                                        polytope[s[0]]):
#                 edges.append((s[1], s[2]))
#             else:
#                 edges.append((s[2], s[0]))
#         else:
#             edges.append((s[1], s[2]))
#             if distance.euclidean(polytope[s[0]], polytope[s[1]]) < distance.euclidean(polytope[s[2]],
#                                                                                        polytope[s[0]]):
#                 edges.append((s[0], s[1]))
#             else:
#                 edges.append((s[2], s[0]))
#         for v0, v1 in edges:
#             ax.plot(xs=polytope[[v0, v1], 0], ys=polytope[[v0, v1], 1], zs=polytope[[v0, v1], 2],
#                     color=color, alpha=0.5)
#     ax.scatter(polytope[:, 0], polytope[:, 1], polytope[:, 2], marker='o', color=color)


if __name__ == "__main__":
    connectivity = np.array([[0, 0, 1],
                             [1, 0, 1],
                             [0, 1, 0]])

    # connectivity = np.array([[1, 1, 0],
    #                          [0, 1, 1],
    #                          [1, 0, 1]])
    graph = generate_graph(connectivity_matrix=connectivity, type="random", size=6, self_loop=True, undirected=True)
    # graph.visualize_graph()

    # Q-prop
    q_prop = QProp(graph=graph)
    fraction_flag = q_prop.multi_stage_prop(steps=10)

    # create canvas
    fig = plt.figure()

    q_set_face_array = np.zeros((6, 3, 6, 5, 3)) - 1
    # plot Q-sets
    colors = ['b', 'y', 'g']
    for t in range(len(q_prop)):
        q_set_t = []
        alpha_min_t = []
        for i in range(3):
            ax = fig.add_subplot(3, len(q_prop), len(q_prop) * i + t + 1, projection='3d')
            faces = plot_polyhedron(vertices=q_prop.Q_flow[t][i].vertices, ax=ax, color=colors[i])
            for j, face in enumerate(faces):
                for k, point in enumerate(face):
                    q_set_face_array[t, i, j, k, :] = point

    mdict = {"data": q_set_face_array}
    io.savemat("3_node_q_prop_example.mat", mdict=mdict)

    plt.show()
