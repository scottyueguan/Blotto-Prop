import numpy as np


class Environment:
    def __init__(self, connectivity_X, Tf, X, Y, connectivity_Y=None):
        self.connectivity_X = connectivity_X
        self.Tf = Tf
        if connectivity_Y is None:
            self.connectivity_Y = connectivity_X
        else:
            self.connectivity_Y = connectivity_Y

        self.X = X
        self.Y = Y

    def set_X(self, X):
        self.X = X

    def set_Y(self, Y):
        self.Y = Y

    def get_dimension_X(self):
        return self.connectivity_X.shape[0]

    def get_dimension_Y(self):
        return self.connectivity_Y.shape[0]


def generate_env_from_name(name: str):
    if name == 'figure-4':
        connectivity = np.array([[1, 1, 0, 0, 1],
                                 [0, 1, 1, 1, 0],
                                 [0, 0, 1, 1, 0],
                                 [0, 0, 1, 1, 1],
                                 [1, 0, 0, 0, 1]])
        Tf = 10

        X = 3.01
        Y = 1

        return Environment(connectivity_X=connectivity, Tf=Tf, X=X, Y=Y)

    elif name == "3-node":
        connectivity = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0]])
        Tf = 5
        X, Y = 3.1, 1
        return Environment(connectivity_X=connectivity, Tf=Tf, X=X, Y=Y)

    elif name == "figure-4-v2":
        connectivity = np.array([[1, 1, 0, 0, 1, 0, 0],  # 1
                                 [0, 1, 1, 1, 0, 0, 0],  # 2
                                 [0, 0, 1, 1, 0, 0, 0],  # 3
                                 [0, 0, 1, 1, 1, 0, 0],  # 4
                                 [0, 0, 0, 0, 1, 1, 0],  # 5
                                 [0, 0, 0, 0, 0, 1, 1],  # 6
                                 [1, 0, 0, 0, 0, 0, 1]])  # 7
        Tf = 10

        X = 4
        Y = 1

        return Environment(connectivity_X=connectivity, Tf=Tf, X=X, Y=Y)

    elif name == "fig-2":
        connectivity = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        Tf = 5
        X, Y = 10, 4
        return Environment(connectivity_X=connectivity, Tf=Tf, X=X, Y=Y)

    else:
        raise Exception("Environment name not valid!")

        # #                         1  2  3  4  5  6  7  8  9  10
        # connectivity = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 1
        #                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 2
        #                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 3
        #                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 4
        #                          [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],  # 5
        #                          [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],  # 6
        #                          [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 7
        #                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # 8
        #                          [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # 9
        #                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]) # 10

        #                         1  2  3  4  5  6  7  8  9  10
