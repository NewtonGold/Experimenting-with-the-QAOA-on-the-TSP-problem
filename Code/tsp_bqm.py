import abc
import numpy as np
import dimod
import os
import dwave.samplers as ds
import networkx as nx
import matplotlib.pyplot as plt
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model

class TSPBQMFormulator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__():
        pass

    @abc.abstractmethod
    def encode_var():
        pass

    @abc.abstractmethod
    def decode_solution():
        pass

    @abc.abstractmethod
    def form_bqm():
        pass

    @abc.abstractmethod
    def convert_solution_to_edge_list():
        pass

    def save_new_bqm(self, tsp_matrix: np.ndarray, path: str, \
                     weight_factor: float = None, penalty_c: float = None) -> \
                        tuple[dimod.BinaryQuadraticModel, float]:
        '''
        Saves a new BQM to a file.

        Parameters
        ----------
        tsp_matrix : np.ndarray
            The adjacency matrix of the graph.
        path : str
            The path to the directory where the BQM will be saved.
        weight_factor : float
            The factor by which the weights of the edges will be multiplied by.
            By default, weight_factor = tsp_matrix.sum() * total_nodes /
             total_edges.
        penalty_c : float
            An additional penalty for the constraints penalty.
            Where penalty_a is the penalty for the constraints,
            and penalty_a = (penalty_b * max_weight) * penalty_c.
            By default, penalty_a = (penalty_b * max_weight) + 1.

        Returns
        -------
        bqm : dimod.BinaryQuadraticModel
            The BQM for the TSP.
        weight_factor : float
            The factor by which the weights of the edges were multiplied by.
        '''
        total_n = len(tsp_matrix)
        bqm, weight_factor = self.form_bqm(tsp_matrix, \
                                           penalty_b=weight_factor, \
                                            penalty_c=penalty_c)
        output_path = self.save_bqm(bqm, total_n, path, weight_factor)

        return bqm, weight_factor, output_path

    def save_bqm(self, bqm: dimod.BinaryQuadraticModel, total_n: int, \
                 path: str, weight_factor: float) -> str:
        path += '_' + str(total_n)
        count = 0
        output_path = path + '_' + str(count)
        while os.path.exists(output_path):
            count += 1
            output_path = path + '_' + str(count)

        # Create the output directory
        os.mkdir(output_path)

        with open(output_path + '/bqm', 'wb') as f:
            f.write(bqm.to_file().read())

        with open(output_path + '/weight_factor', 'w') as f:
            f.write(str(weight_factor))

        return output_path

    def load_bqm(self, path: str):
        with open(path + '/bqm', 'rb') as f:
            bqm = dimod.BinaryQuadraticModel.from_file(f)

        with open(path + '/weight_factor', 'r') as f:
            weight_factor = float(f.read())

        return bqm, weight_factor

    def generate_tsp_matrix(self, total_n: int, p: float, \
                            weights: tuple[float, float] = (1, 10)) \
                                -> np.ndarray:
        '''
        Generates a random adjacency matrix for a TSP graph.

        Parameters
        ----------
        total_n : int
            The number of nodes in the graph.
        p : float
            The probability of an edge existing between two nodes.
        weights : tuple[float, float]
            The range of weights for the edges. By default, the weights
            are integers between 1 and 10.

        Returns
        -------
        tsp_matrix : np.ndarray
            The adjacency matrix of the graph.
        '''
        if weights[0] > weights[1]:
            raise ValueError('The lower bound of the weights must be less than'
                             'the upper bound. The lower bound was {} and the '
                             'upper bound was {}.'.format(weights[0], \
                                                          weights[1]))
        elif weights[0] < 1:
            raise ValueError('The lower bound of the weights must be greater '
                             'than 0. The lower bound was {}.'\
                                .format(weights[0]))

        tsp_matrix = np.zeros((total_n, total_n), dtype=int)
        for i in range(total_n):
            for j in range(i+1, total_n):
                if np.random.rand() < p:
                    rand_weight = np.random.randint(weights[0], weights[1])
                    tsp_matrix[i, j] = rand_weight
                    tsp_matrix[j, i] = rand_weight
        return tsp_matrix
    
    def bqm_to_coefficient_array(self, bqm: dimod.BinaryQuadraticModel) -> \
        np.ndarray:
        '''
        Convert a binary quadratic model to a coefficient array

        Parameters
        ----------
        bqm : dimod.BinaryQuadraticModel
            The binary quadratic model to be converted

        Returns
        -------
        np.ndarray
            The coefficient array
        '''

        variables = sorted(bqm.variables)
        num_variables = len(variables)

        coefficient_array = np.zeros((num_variables, num_variables))

        for (i, j), value in bqm.quadratic.items():
            coefficient_array[variables.index(i), variables.index(j)] = value

        for i, value in bqm.linear.items():
            coefficient_array[variables.index(i), variables.index(i)] = value

        return coefficient_array
    
    def solve_bqm(self, bqm: dimod.BinaryQuadraticModel, \
                sampler: dimod.Sampler = ds.SimulatedAnnealingSampler()) \
                    -> dimod.SampleSet:
        '''
        Solve the BQM using the given sampler

        Parameters
        ----------
        bqm : dimod.BinaryQuadraticModel
            The binary quadratic model to be solved
        sampler : dimod.Sampler, optional
            The sampler to be used, by default ds.SimulatedAnnealingSampler()

        Returns
        -------
        dimod.SampleSet
            The sampleset containing the best solution
        '''

        # # Solve the problem
        sampleset = sampler.sample(bqm)

        # # Store the best solution
        best_solution = sampleset.first.sample

        return best_solution
    
    def get_tour(self, solution: dict, total_n: int) -> list:
        '''
        Given a solution return the order of the nodes

        Parameters
        ----------
        solution : dict
            The solution to the QUBO
        total_n : int
            The total number of nodes
        
        Returns
        -------
        list
            The order of the nodes
        '''

        # # Extract the edges that appear in the path
        path_edges = self.convert_solution_to_edge_list(solution, total_n)

        # # Find the nodes that appear in the path
        path_nodes = []
        current_node = 0
        for _ in range(total_n + 1):
            path_nodes.append(current_node)
            for edge in path_edges:
                if edge[0] == current_node:
                    current_node = edge[1]
                    break

        return path_nodes
    
    def tour_cost(self, solution: dict, tsp_matrix: np.ndarray) -> float:
        '''
        Calculate the cost of a tour given an adjacency matrix
        
        Parameters
        ----------
        solution : dict
            The solution to the QUBO
        tsp_matrix : np.ndarray
            The adjacency matrix of the TSP problem
        
        Returns
        -------
        float
            The cost of the route
        '''
        total_n = len(tsp_matrix)
        tour_edges = self.convert_solution_to_edge_list(solution, total_n)

        cost = 0
        for edge in tour_edges:
            cost += tsp_matrix[edge[0]][edge[1]]

        return cost
    
    def graph_solution(self, solution: dict, tsp_array: np.ndarray):
        '''
        Graph the solution to the TSP problem

        Parameters
        ----------
        solution : dict
            The solution to be graphed
        tsp_array : np.ndarray
            The adjacency matrix of the TSP problem
        '''


        # # Create a graph from the TSP array
        G = nx.from_numpy_array(tsp_array)

        # # Create a list of edges in the solution tour
        total_n = len(tsp_array)
        edges = self.convert_solution_to_edge_list(solution, total_n)

        # # Create a subgraph from the edges in the solution tour
        H = G.edge_subgraph(edges)

        # # Draw the initial graph and the solution tour
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        nx.draw(H, pos, edge_color='r', width=2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): tsp_array[i, j] \
                                                        for i, j in G.edges()})
        plt.show()

    def convert_bqm_to_quadratic_program(self, bqm):

        # # Create a Model
        mdl = Model()

        # # Add variables to the Model
        x = mdl.binary_var_list(bqm.variables, name='x')

        objective_function = 0

        # # Add the linear terms to the Model
        for v, c in bqm.linear.items():
            objective_function += c * x[v]

        # # Add the quadratic terms to the Model
        for (u, v), c in bqm.quadratic.items():
            objective_function += c * x[u] * x[v]

        # # Add the offset to the Model
        objective_function += bqm.offset

        # # Set the objective function
        mdl.minimize(objective_function)

        # # Covert the Model to a QuadraticProgram
        qp = from_docplex_mp(mdl)

        # # Return the QuadraticProgram
        return qp

