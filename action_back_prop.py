import numpy as np
import gurobipy as gp
from gurobipy import GRB


class BackProp:
    def __init__(self):
        self.X_ir_list = None
        self.b = None
        self.len_i, self.len_r = None, None
        self.x_shape = None
        self.done_flag = False

    def set_target(self, b):
        self.b = b

    def set_current_set(self, vertices, extreme_actions):
        self.len_i = len(extreme_actions)
        self.len_r = len(vertices)
        self.x_shape = vertices[0].shape
        self.done_flag = False

        self.X_ir_list = []
        for i in range(self.len_i):
            X_i_list = []
            for r in range(self.len_r):
                X_i_list.append(np.matmul(vertices[r].T, extreme_actions[i]))
            self.X_ir_list.append(X_i_list)

    def solve(self):
        # create model
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as m:
                # create variables
                lambdas = [m.addVar(name='lambda_{}'.format(i)) for i in range(self.len_i)]
                thetas = [m.addVar(name='theta_{}'.format(r)) for r in range(self.len_r)]

                # create objective:
                m.setObjective(0, GRB.MAXIMIZE)

                # add bilinear equality constraints
                for j in range(self.x_shape[0]):
                    m.addConstr(sum([lambdas[i] * thetas[r] * self.X_ir_list[i][r][0, j] for i in range(self.len_i) for r in
                                     range(self.len_r)]) == self.b[0, j])

                # add linear equality constraints
                m.addConstr(sum(lambdas) == 1)
                m.addConstr(sum(thetas) == 1)

                # set lower bounds
                for lmbd in lambdas:
                    lmbd.lb = 0
                for theta in thetas:
                    theta.lb = 0

                # for debugging print out the problem constraints
                # m.write("file.lp")

                m.Params.NonConvex = 2
                m.optimize()
                # m.printAttr('x')
                lambda_solution = [lambdas[i].x for i in range(len(lambdas))]
                theta_solution = [thetas[i].x for i in range(len(thetas))]

        return lambda_solution, theta_solution


if __name__ == '__main__':
    x1 = np.array([1, 1, 0]).reshape((-1, 1))
    x2 = np.array([0.5, 0.5, 1]).reshape((-1, 1))
    x3 = np.array([1.5, 0, 0.5]).reshape((-1, 1))

    K1 = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]])
    K2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    lamb1 = 0.6
    lamb2 = 0.4
    theta1 = 0.2
    theta2 = 0.5
    theta3 = 0.3

    K = lamb1 * K1 + lamb2 * K2
    x = theta1 * x1 + theta2 * x2 + theta3 * x3

    b = np.matmul(x.T, K)
    print(b)

    backprop = BackProp()
    backprop.set_current_set(vertices=[x1, x2, x3], extreme_actions=[K1, K2])
    backprop.set_target(b=b)
    lambdas, thetas = backprop.solve()

    K_solved = K1 * lambdas[0] + K2 * lambdas[1]
    x_solved = x1 * thetas[0] + x2 * thetas[1] + x3 * thetas[2]

    print(np.matmul(x.T, K))
