import numpy as np
import gurobipy as gp
from gurobipy import GRB


class BackProp:
    def __init__(self, X_ir, b):
        self.X_ir_list = X_ir
        self.b = b
        self.len_i, self.len_r = len(X_ir), len(X_ir[0])
        self.x_shape = (X_ir[0][0]).shape
        self.done_flag = False

    def solve(self):
        # create model
        m = gp.Model('bilinear')

        # create variables
        lambdas = [m.addVar(name='lambda_{}'.format(i)) for i in range(self.len_i)]
        thetas = [m.addVar(name='theta_{}'.format(r)) for r in range(self.len_r)]

        # create objective:
        m.setObjective(0, GRB.MAXIMIZE)

        # add bilinear equality constraints
        for j in range(self.x_shape[0]):
            m.addConstr(sum([lambdas[i] * thetas[r] * self.X_ir_list[i][r][j] for i in range(self.len_i) for r in range(self.len_r)]) == self.b[j])

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
    x1 = np.array([1, 1]).reshape((-1, 1))
    x2 = np.array([-1, 3]).reshape((-1, 1))
    x3 = np.array([2, 1]).reshape((-1, 1))
    x4 = np.array([0, -1]).reshape((-1, 1))

    X_ir = [[x1, x2], [x3, x4]]

    lamb1 = 0.6
    lamb2 = 0.4
    theta1 = 0.2
    theta2 = 0.8

    b = lamb1 * theta1 * x1 + lamb1 * theta2 * x2 + lamb2 * theta1 * x3 + lamb2 * theta2 * x4

    backprop = BackProp(X_ir=X_ir, b=b)
    lambdas, thetas = backprop.solve()

    print(lambdas)
    print(thetas)
