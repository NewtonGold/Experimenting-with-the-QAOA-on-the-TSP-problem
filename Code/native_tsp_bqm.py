import numpy as np
import dimod
import tsp_bqm

class NativeBQMFormulator(tsp_bqm.TSPBQMFormulator):

    def __init__(self) -> None:
        pass

    def encode_var(self, i, j, total_n):
        # # Define a function to map node indices to binary variables
        return (i*total_n) + j

    def decode_solution(self, solution, total_n):
        # # Define a function to map binary variables to node indices
        new_solution = {}
        for i, item in enumerate(solution.items()):
            new_solution[(i)] = item[1]

        decoded_solution = {}
        count = 0
        for i in range(total_n):
            for j in range(total_n):
                if i == 0 and j == 0:
                    decoded_solution[(i, j)] = 1
                elif i == 0 or j == 0:
                    decoded_solution[(i, j)] = 0
                else:
                    decoded_solution[(i, j)] = new_solution[count]
                    count += 1

        return decoded_solution

    def form_bqm(self, tsp_matrix: np.ndarray, penalty_b: int = None, \
                 penalty_c: float = None):
        '''
            Creates the hamiltonian for the TSP
        '''
        # # Number of nodes in the graph
        total_n = len(tsp_matrix)

        # # Number of edges in the graph
        total_edges = 0
        for i in range(len(tsp_matrix)):
            for j in range(len(tsp_matrix)):
                if tsp_matrix[i, j] != 0:
                    total_edges += 1

        # # If no penalty is given, set it to average tour length.
        # # This was the method used by D-Wave in their implementation of the 
        # #native TSP.
        if penalty_b == None:
            penalty_b = tsp_matrix.sum()*total_n/total_edges

        # # A is a large enough constant to make sure that the constraints are 
        # # satisfied.
        # # I have calculated the value for A in the same way as in the native 
        # # TSP.
        if penalty_c == None:
            penalty_a = (penalty_b * tsp_matrix.max()) + 1
        else:
            penalty_a = (penalty_b * tsp_matrix.max()) * penalty_c

        x = list(dimod.Binaries([(i) for i in range(total_n**2)]))

        h_distance = self.distance_objective(tsp_matrix, x, total_n)
        h_c_1 = self.constraint_1(x, total_n)
        h_c_2 = self.constraint_2(x, total_n)
        h_c_3 = self.constraint_3(tsp_matrix, x, total_n)

        h_final = penalty_a * (h_c_1 + h_c_2 + h_c_3) + penalty_b * h_distance

        for i in range(total_n):
            for j in range(total_n):
                if i == 0 and j == 0:
                    h_final.fix_variable(self.encode_var(i, j, total_n), 1)
                elif i == 0 or j == 0:
                    h_final.fix_variable(self.encode_var(i, j, total_n), 0)

        # Get the variable labels
        labels = h_final.variables

        # Create a mapping from variable labels to linear indices
        mapping = {label: i for i, label in enumerate(labels)}

        # Relabel the variables in the BQM
        linear_bqm = h_final.relabel_variables(mapping, inplace=False)


        return linear_bqm, penalty_b
    
    def distance_objective(self, tsp_matrix: np.ndarray, \
                        x: dict[(int), dimod.Binary], \
                            total_n: int) -> dimod.BinaryQuadraticModel:
        '''
        Distance objective function.

        Parameters
        ----------
        tsp_matrix : np.ndarray
            The matrix of the TSP.
        x : dict[(int, int, int), dimod.Binary]
            The variables of the QUBO.
        total_n : int
            The number of nodes in the TSP.

        Returns
        -------
        h_distance : dimod.BinaryQuadraticModel
            The distance objective function of the QUBO.
        '''
        h_distance = 0
        for u in range(total_n):
            for v in range(total_n):
                temp_h_distance = 0
                for j in range(total_n):
                    if tsp_matrix[u][v] != 0:
                        if j == total_n-1:
                            temp_h_distance += \
                                x[self.encode_var(u, j, total_n)] * \
                                    x[self.encode_var(v, 0, total_n)]
                        else:
                            temp_h_distance += \
                                x[self.encode_var(u, j, total_n)] * \
                                    x[self.encode_var(v, j+1, total_n)]
                h_distance += tsp_matrix[u][v] * temp_h_distance

        return h_distance
    
    def constraint_1(self, x: dict[(int), dimod.Binary], \
                            total_n: int) -> dimod.BinaryQuadraticModel:
        '''
        Constraint so that every vertex can only appear once in the solution 
        cycle.

        Parameters
        ----------
        x : dict[(int), dimod.Binary]
            The variables of the QUBO.
        total_n : int
            The number of nodes in the TSP.

        Returns
        -------
        h_c_1 : dimod.BinaryQuadraticModel
            Constraint 1 of the QUBO.
        '''
        h_c_1 = 0
        for v in range(total_n):
            temp = 0
            for j in range(total_n):
                temp += x[self.encode_var(v, j, total_n)]
            h_c_1 += (1 - temp) ** 2
        
        return h_c_1

    def constraint_2(self, x: dict[(int), dimod.Binary], \
                            total_n: int) -> dimod.BinaryQuadraticModel:
        '''
        For every potential j value in the solution there must be a 
        corresponding jth node.

        Parameters
        ----------
        x : dict[(int), dimod.Binary]
            The variables of the QUBO.
        total_n : int
            The number of nodes in the TSP.

        Returns
        -------
        h_c_2 : dimod.BinaryQuadraticModel
            Constraint 2 of the QUBO.

        '''
        h_c_2 = 0
        for j in range(total_n):
            temp = 0
            for v in range(total_n):
                temp += x[self.encode_var(v, j, total_n)]
            h_c_2 += (1-temp) ** 2

        return h_c_2
        

    def constraint_3(self, tsp_matrix: np.ndarray, \
                        x: dict[(int), dimod.Binary], \
                            total_n: int) -> dimod.BinaryQuadraticModel:
        '''
        Constraint so that subsequent nodes in the solution cycle are connected
        by an edge.

        Parameters
        ----------
        tsp_matrix : np.ndarray
            The matrix of the TSP.
        x : dict[(int), dimod.Binary]
            The variables of the QUBO.
        total_n : int
            The number of nodes in the TSP.

        Returns
        -------
        h_c_3 : dimod.BinaryQuadraticModel
            Constraint 3 of the QUBO.
        '''
        h_c_3 = 0
        for u in range(total_n):
            for v in range(total_n):
                for j in range(total_n):
                    if tsp_matrix[u][v] == 0:
                        if j == total_n-1:
                            h_c_3 += x[self.encode_var(u, j, total_n)] * \
                                x[self.encode_var(v, 0, total_n)]
                        else:
                            h_c_3 += x[self.encode_var(u, j, total_n)] * \
                                x[self.encode_var(v, j+1, total_n)]

        return h_c_3
    
    def find_invalid_constraints(self, solution, total_n, tsp_matrix) -> \
        tuple[list, list, list]:
        '''
        Given a solution for each constraint create a list of edges that break the 
        constraint.


        Parameters
        ----------
        solution : dict
            The solution to the QUBO.
        total_n : int
            The total number of nodes.

        Returns
        -------
        c_1 : list
            List of variables that break constraint 1.
        c_2 : list
            List of variables that break constraint 2.
        c_3 : list
            List of variables that break constraint 3.
        '''
        decoded_solution = self.decode_solution(solution, total_n)
        c_1 = []
        for v in range(total_n):
            temp = 0
            for j in range(total_n):
                temp += decoded_solution[(v, j)]
            if temp != 1:
                c_1.append(v)

        c_2 = []
        for j in range(total_n):
            temp = 0
            for v in range(total_n):
                temp += decoded_solution[(v, j)]
            if temp != 1:
                c_2.append(j)

        c_3 = []
        for u in range(total_n):
            for v in range(total_n):
                for j in range(total_n):
                    if tsp_matrix[u][v] == 0:
                        if j == total_n-1:
                            if decoded_solution[(u, j)] == 1 and \
                            decoded_solution[(v, 0)] == 1:
                                c_3.append((u, v))
                        elif decoded_solution[(u, j)] == 1 and \
                            decoded_solution[(v, j+1)] == 1:
                            c_3.append((u, v))

        return c_1, c_2, c_3
    
    def validity_score(self, solution: dict, total_n: int, \
                       tsp_matrix: np.ndarray) -> tuple[dict, list]:
        '''
        Calculate the validity score of a solution and a list of which constraints 
        are broken.

        Parameters
        ----------
        solution : dict
            The solution to the QUBO
        total_n : int
            The total number of nodes
        tsp_matrix : np.ndarray
            The matrix of the TSP

        Returns
        -------
        int
            The number of broken constraints
        list
            The list of broken constraints
        '''
        # # Uses the find invalid constraints function to determine a score of how 
        # # many constraints are broken.
        constraints = self.find_invalid_constraints(solution, total_n, \
                                                    tsp_matrix)
        score = 0
        broken_constraints = []
        for i, constraint in enumerate(constraints):
            if len(constraint) > 0:
                score += 1
                broken_constraints.append(i + 1)

        return score, broken_constraints

    def convert_solution_to_edge_list(self, solution: dict, total_n: int) -> list:
        decoded_solution = self.decode_solution(solution, total_n)
        edge_list = []
        current_time = 0
        current_node = None
        first_node = None
        node_order = [node for node in decoded_solution.keys() if \
                      decoded_solution[node] == 1]
        
        for i in range(total_n + 1):
            for node in node_order:
                if i == total_n and current_node != None:
                    edge_list.append((current_node, first_node))
                    break
                if node[1] == current_time:
                    if current_node == None:
                        first_node = node[0]
                        current_node = node[0]
                    else:
                        edge_list.append((current_node, node[0]))
                        current_node = node[0]
                    current_time += 1
                    break

        return edge_list

    def tour_to_solution(self, node_order):
        total_n = len(node_order)
        solution = {}
        for i in range(total_n):
            for j in range(i+1, total_n):
                u = node_order[i]
                v = node_order[j]
                if i == 0 and j == total_n-1:
                    solution[(u, v)] = 1
                elif j == i+1:
                    solution[(u, v)] = 1
                else:
                    solution[(u, v)] = 0
        return solution

def main():
    # fname = "TSP_Instances/tsp_5_2"
    fname = "TSP_Instances/tsp_5_3"
    tsp_matrix = np.loadtxt(fname)
    tsp_qubo = NativeBQMFormulator()
    # tsp_matrix = tsp_qubo.generate_tsp_matrix(5, 1)
    total_n = len(tsp_matrix)
    bqm, penalty_b = tsp_qubo.form_bqm(tsp_matrix)
    import dwave.samplers as ds
    sampler = ds.SimulatedAnnealingSampler()
    sampler.parameters = {'num_reads': 100}
    best_solution = tsp_qubo.solve_bqm(bqm, sampler=sampler)
    print('Best Solution: ', best_solution)
    print('Decoded Solution: ', tsp_qubo.decode_solution(best_solution, \
                                                         total_n))
    print('Best Energy: ', bqm.energy(best_solution)/penalty_b)

    route = tsp_qubo.get_tour(best_solution, total_n)
    print('Route: ', route)

    cost = tsp_qubo.tour_cost(best_solution, tsp_matrix)
    print('Cost: ', cost)

    edge_solution = tsp_qubo.convert_solution_to_edge_list(best_solution, total_n)
    print('Edge Solution: ', edge_solution)

    tsp_qubo.graph_solution(best_solution, tsp_matrix)

def create_tsp_instance():
    tsp_qubo = NativeBQMFormulator()
    np.savetxt("TSP_Instances/tsp_5_4", tsp_qubo.generate_tsp_matrix(5, 1, (1, 100)))

    

if __name__ == '__main__':
    # create_tsp_instance()
    main()