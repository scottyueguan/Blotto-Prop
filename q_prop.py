import numpy as np
from utils.utils import Vertices, generate_x_req_set, compare_vertices, isSamePoint, isInteger
from convex_hull_algs import convex_hull, con2vert, isInHull
from copy import deepcopy
from typing import List
from graph_generators import generate_graph, Graph
from tqdm import tqdm


class QProp:
    def __init__(self, graph: Graph, T=50, resolution=0.2, eps=0, hull_method="aux_point",
                 need_connections=False, check_N_degenerate=False):
        self.graph = graph
        connectivity = deepcopy(graph.connectivity)
        self.connectivity = connectivity
        self.reverse_connectivity = np.array(connectivity).T
        self.N = len(self.connectivity)
        self.T = T

        self.degenerate_flag = False
        self.check_N_degenerate = check_N_degenerate

        self.neighbor_nodes = self._generate_neighbor_nodes()

        self.extreme_actions = self._generate_extreme_actions(connectivity)
        self.reverse_extreme_actions = self._generate_extreme_actions(self.reverse_connectivity)
        self.extreme_attacker_allocations = self._generate_extreme_attacker_allocations()
        self.Preq_list = self._generate_Preq()
        self.Q_flow = [deepcopy(self.Preq_list)]

    def __len__(self):
        return len(self.Q_flow)

    def _degenerate(self):
        self.degenerate_flag = True
        print("Degenerate case, requires Defender resource on all nodes.")

    def backprop(self):
        if self.degenerate_flag:
            pass
        else:
            Q_k = []
            info = dict()
            fraction_flag = False

            reverse_reachable_set_list = []
            for i in tqdm(range(self.N)):
                if not self.degenerate_flag:
                    neighbor_q_set = self.Q_flow[-1][i]
                    reverse_reachable_set = self.generate_reachable_set(neighbor_q_set.vertices, reverse=True)
                    reverse_reachable_set_list.append(reverse_reachable_set)

            for i in tqdm(range(self.N)):
                if not self.degenerate_flag:
                    neighbor_nodes_i = self.neighbor_nodes[i]

                    constraints_A = [reverse_reachable_set_list[neighbor_node].equations["A"]
                                     for neighbor_node in neighbor_nodes_i]
                    constraints_b = [reverse_reachable_set_list[neighbor_node].equations["b"]
                                     for neighbor_node in neighbor_nodes_i]

                    constraints_A.append(self.Preq_list[i].equations["A"])
                    constraints_b.append(self.Preq_list[i].equations["b"])

                    new_A = np.concatenate(tuple(constraints_A), axis=0)
                    new_b = np.concatenate(tuple(constraints_b), axis=0)

                    vertices, rays, found = con2vert(new_A, new_b)
                    assert found

                    for vertex in vertices:
                        if not isInteger(vertex):
                            print(vertex)
                            fraction_flag = True

                    Q_k.append(vertices)

            self.Q_flow.append(Q_k)

            info["fraction"] = fraction_flag

            return info

    def multi_stage_prop(self, steps):
        step = 0
        fraction_flag = False

        if self.degenerate_flag:
            return False

        print("Starting back prop...")
        while step < steps and not self.degenerate_flag \
                and (self.__len__() < 2 or (self.__len__() >= 2 and not self.isConsistent())):
            print("{} step out of {} steps".format(step, steps))
            info = self.backprop()
            step += 1
            if info["fraction"]:
                fraction_flag = True

        return fraction_flag

    def isConsistent(self):
        consistent_flag = True
        for i in range(self.N):
            vertices_new, vertices_old = self.Q_flow[-1][i], self.Q_flow[-2][i]
            if not compare_vertices(vertices_new, vertices_old):
                consistent_flag = False
                break
        if consistent_flag:
            print("Fixed point found!")
        return consistent_flag

    def _generate_Preq(self):
        Preq_list = []
        extreme_attacker_reachable_vertices = self._generate_extreme_attacker_reachable_vertices()
        for i in range(self.N):
            y_vertices = extreme_attacker_reachable_vertices[i]
            Preq_i, degenerate = generate_x_req_set(vertices_y=y_vertices, X=None)
            Preq_list.append(Preq_i)
            if degenerate:
                self._degenerate()
                return None
        return Preq_list

    def _generate_extreme_attacker_reachable_vertices(self):
        extreme_vertices_list = []
        for i in range(self.N):
            new_vertices_index_list = self.neighbor_nodes[i]
            new_vertices_list = []
            for new_vertices_index in new_vertices_index_list:
                new_vertex = np.zeros(self.N)
                new_vertex[new_vertices_index] = 1.0
                new_vertices_list.append(new_vertex)
            new_vertices = Vertices(new_vertices_list)
            extreme_vertices_list.append(new_vertices)
        return extreme_vertices_list

    def _generate_extreme_attacker_allocations(self):
        extreme_attacker_allocations = []
        for i in range(self.N):
            vertex = np.zeros(self.N)
            vertex[i] = 1.0
            extreme_attacker_allocation = Vertices([vertex])
            extreme_attacker_allocations.append(extreme_attacker_allocation)
        return extreme_attacker_allocations

    def generate_reachable_set(self, x_points: List, reverse=False):
        new_vertices = []
        if not reverse:
            extreme_actions = self.extreme_actions
        else:
            extreme_actions = self.reverse_extreme_actions

        for extreme_actions in extreme_actions:
            for x_point in x_points:
                new_vertices.append(np.matmul(x_point, extreme_actions))

        if all([abs(np.sum(vertex) - self.N) < 1e-10 for vertex in new_vertices]) and self.check_N_degenerate:
            self._degenerate()
            return None

        polytope, success = convex_hull(new_vertices, need_equations=True)
        assert success

        return polytope

    def _generate_neighbor_nodes(self):
        neighbor_nodes_list = []
        for i in range(self.N):
            neighbor_nodes = []
            for j in range(self.N):
                if self.connectivity[i, j] > 0:
                    neighbor_nodes.append(j)
            neighbor_nodes_list.append(neighbor_nodes)
        return neighbor_nodes_list

    def _generate_extreme_actions(self, connectivity):
        return self._expand(0, list=[np.array([])], connectivity_matrix=connectivity)

    def _expand(self, n, list, connectivity_matrix):
        n_children = sum(connectivity_matrix[n, :])
        non_zero_indices = np.nonzero(connectivity_matrix[n, :])[0]

        for i in range(len(list)):
            current_list = list.pop(0)

            for j in range(n_children):
                new_row = np.zeros(self.N)
                new_row[non_zero_indices[j]] = 1
                if n > 0:
                    new_action = np.vstack((current_list, new_row))
                else:
                    new_action = new_row
                list.append(new_action)

        if n == self.N - 1:
            return list
        else:
            list = self._expand(n + 1, list, connectivity_matrix)

        return list


if __name__ == "__main__":
    # connectivity = np.array([[1, 1, 0, 0, 1],
    #                          [0, 1, 1, 1, 0],
    #                          [0, 0, 1, 1, 0],
    #                          [0, 0, 1, 1, 1],
    #                          [1, 0, 0, 0, 1]])

    connectivity = np.array([[0, 0, 1],
                             [1, 0, 1],
                             [0, 1, 0]])

    # connectivity = np.array([[1, 0, 0, 1, 0, 0],
    #                          [1, 0, 0, 0, 0, 0],
    #                          [0, 0, 1, 0, 0, 0],
    #                          [0, 0, 0, 1, 1, 0],
    #                          [0, 0, 0, 0, 1, 1],
    #                          [1, 0, 0, 0, 1, 0]])

    # connectivity = np.array([[0, 0, 1, 1, 0, 1],
    #                          [0, 0, 0, 0, 1, 1],
    #                          [0, 1, array([1, 0, 1, 0],
    #                          [1, 0, 0, 0, 1, 0],
    #                          [0, 1, 1, 0, 1, 0],
    #                          [1, 0, 0, 0, 1, 1]])

    # graph = generate_graph(connectivity_matrix=None, type="random", size=6, self_loop=True, undirected=True)
    graph = generate_graph(connectivity_matrix=connectivity, type=None, size=6, self_loop=True, undirected=True)
    # graph.visualize_graph()

    q_prop = QProp(graph=graph)
    fraction_flag = q_prop.multi_stage_prop(steps=10)
    for t in range(len(q_prop)):
        alpha_min_t = []
        for i in range(graph.connectivity.shape[0]):
            alpha_i = []
            for vertex in q_prop.Q_flow[t][i].vertices:
                alpha_i.append(np.sum(vertex))
            alpha_i_min = np.min(alpha_i)
            alpha_min_t.append(np.round_(alpha_i_min, decimals=2))
        print(alpha_min_t)

    i = 0


    print("done!")
