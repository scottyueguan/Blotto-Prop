import numpy as np


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
                connectivity[i,:i] = connectivity.T[i, :i]

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
        return ring_graph_generator(size, **kwargs)
    if type == "random":
        return random_graph_generator(size, **kwargs)

    else:
        raise Exception("type error!")


if __name__ == "__main__":
    connectivity = generate_graph(type="random", size=10, undirected=True, self_loop=False)

    print(connectivity)