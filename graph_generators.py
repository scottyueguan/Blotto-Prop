import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, connectivity):
        self.connectivity = connectivity
        self.N = connectivity.shape[0]

    def visualize_graph(self):
        G = nx.DiGraph()
        for i in range(self.N):
            for j in range(self.N):
                if self.connectivity[i, j] > 0:
                    G.add_edge(str(i), str(j))
        nx.draw(G, arrows=True, with_labels=True, node_color='w', edgecolors='k')
        plt.show()


def ring_graph_generator(size, undirected=True):
    connectivity = np.zeros((size, size))
    if undirected:
        for i in range(size):
            if i == 0:
                connectivity[i, i] = 1
                connectivity[i, i + 1] = 1
                connectivity[i, -1] = 1
            elif i == size - 1:
                connectivity[i, i] = 1
                connectivity[i, 0] = 1
                connectivity[i, i - 1] = 1

            else:
                connectivity[i, i] = 1
                connectivity[i, i - 1] = 1
                connectivity[i, i + 1] = 1
    else:
        for i in range(size):
            if i == size - 1:
                connectivity[i, i] = 1
                connectivity[i, 0] = 1
            else:
                connectivity[i, i] = 1
                connectivity[i, i + 1] = 1

    return connectivity.astype(int)


def random_graph_generator(size, self_loop=False, undirected=False):
    n_nodes = size
    connectivity = np.zeros((size, size))

    done_flag = False

    while not done_flag:
        if not undirected:
            for i in range(n_nodes):
                random_vec = np.random.random(size)
                connectivity[i, :] = np.array([1 if ele < 0.2 else 0 for ele in random_vec])
        else:
            for i in range(n_nodes):
                random_vec = np.random.random((size - i))
                connectivity[i, i:] = np.array([1 if ele < 0.2 else 0 for ele in random_vec])
            for i in range(n_nodes):
                connectivity[i, :i] = connectivity.T[i, :i]

        if self_loop:
            for i in range(n_nodes):
                connectivity[i, i] = 1

        done_flag = True
        for i in range(size):
            if np.sum(connectivity[i, :]) == size or np.sum(connectivity[i, :]) == 0:
                done_flag = False
                break

        for j in range(size):
            if np.sum(connectivity[:, j]) == size or np.sum(connectivity[:, j]) == 0:
                done_flag = False
                break

    return connectivity.astype(int)


def generate_graph(type: str, size: int, **kwargs):
    if type == "ring":
        connectivity = ring_graph_generator(size, **kwargs)
        graph = Graph(connectivity)
        return graph
    if type == "random":
        connectivity = random_graph_generator(size, **kwargs)
        graph = Graph(connectivity)
        return graph

    else:
        raise Exception("type error!")


if __name__ == "__main__":
    graph = generate_graph(type="random", size=10, undirected=False, self_loop=False)
    graph.visualize_graph()
    print(graph.connectivity)
