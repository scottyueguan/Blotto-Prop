import numpy as np
from graph_generators import generate_graph
from matplotlib import pyplot as plt
from q_prop import QProp
from scipy import io
from utils.plot_polyhedron import plot_polyhedron


if __name__ == "__main__":
    connectivity = np.array([[1, 0, 0],
                             [1, 0, 0],
                             [0, 1, 1]])

    graph = generate_graph(connectivity_matrix=connectivity, type="random", size=6, self_loop=True, undirected=True)

    # Q-prop
    q_prop = QProp(graph=graph)
    fraction_flag = q_prop.multi_stage_prop(steps=5)
    T = len(q_prop)
    # create canvas
    fig = plt.figure()

    q_set_face_array = np.zeros((T, 3, 6, 5, 3)) - 1
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
    io.savemat("sink_q_prop_example.mat", mdict=mdict)

    plt.show()
