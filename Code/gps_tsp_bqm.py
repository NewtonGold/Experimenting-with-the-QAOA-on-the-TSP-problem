import numpy as np
import dimod
import tsp_bqm

class GPSBQMFormulator(tsp_bqm.TSPBQMFormulator):

    def __init__(self, R: int = 3):
        '''
        Parameters
        ----------
        R : int, optional
            The number of values r can take. The default is 3.

        Returns
        -------
        None.
        '''

        self.R = R

    def encode_var(self, i: int, j: int, r: int, total_n: int) -> int:
        '''
        A function to map node indices to binary variables.

        Parameters
        ----------
        i : int
            The first node index.
        j : int
            The second node index.
        r : int
            The value of r.
        total_n : int
            The total number of nodes in the graph.

        Returns
        -------
        int
            The encoded single integer variable index.
        '''
        if i < j:
            return i * total_n * self.R + j * self.R + r - (self.R * (i + 1))
        elif i > j:
            return i * total_n * self.R + j * self.R + r - (self.R * i)
        else:
            raise ValueError('i and j cannot be equal.')


    def decode_solution(self, solution: dict, total_n: int) -> dict:
        '''
        A function to map binary variables to node indices.

        Parameters
        ----------
        solution : dict
            The solution dictionary.
        total_n : int
            The total number of nodes in the graph.

        Returns
        -------
        solution : dict
            The solution dictionary with the keys decoded to tuples.
        '''
        solution_copy = solution.copy()
        decoded_solution = {}
        for i in range(total_n):
            for j in range(total_n):
                if i != j:
                    for r in range(self.R):
                        decoded_solution[(i, j, r)] = solution_copy.pop(\
                            self.encode_var(i, j, r, total_n))

        return decoded_solution

    def form_bqm(self, tsp_matrix: np.ndarray, penalty_b: int = None, \
                 penalty_c: float = None) -> dimod.BinaryQuadraticModel:
        '''
        Creates the hamiltonian for the TSP.

        Parameters
        ----------
        tsp_matrix : np.ndarray
            The adjacency matrix of the graph.
        penalty : int, optional
            The penalty to be used in the constraints. The default is None.
            If no penalty is given, it is set to the average tour length.
            Calculated using the following formula:
            penalty = tsp_matrix.sum()*total_n/total_edges

        Returns
        -------
        h_final : dimod.BinaryQuadraticModel
            The hamiltonian for the TSP.
        '''

        # # Number of nodes in the graph
        total_n = len(tsp_matrix)

        # # Number of edges in the graph
        total_edges = 0
        for i in range(total_n):
            for j in range(total_n):
                if tsp_matrix[i, j] != 0:
                    total_edges += 1

        # # If no penalty is given, set it to average tour length.
        # # This was the method used by D-Wave in their implementation of the native
        # # TSP so I have made the assumption that as a default value it should work 
        # # relatively well here too.
        if penalty_b == None:
            penalty_b = tsp_matrix.sum()*total_n/total_edges


        # # A is a large enough constant to make sure that the constraints are 
        # # satisfied.
        # # I have calculated the value for A in the same way as in the native TSP
        if penalty_c == None:
            penalty_a = (penalty_b * tsp_matrix.max()) + 1
        else:
            penalty_a = (penalty_b * tsp_matrix.max()) * penalty_c

        # # Creates a dictionary of variables x_i_j_r
        x = self.generate_variables(total_n)

        # # Distance objective function
        h_distance = self.distance_objective(tsp_matrix, x, total_n)

        # # Constraint 1: For each i, j only 1 r value must be given
        h_c_1 = self.constraint_1(x, total_n)

        # # Constraint 2: Each node must be exited once
        h_c_2 = self.constraint_2(x, total_n)

        # # Constraint 3: Each node must be reached once
        h_c_3 = self.constraint_3(x, total_n)

        # # Constraint 4: If node i is reached before j, 
        # # then node j is reached after i
        h_c_4 = self.constraint_4(x, total_n)

        # # Constraint 5: Prevent subtours and cycles
        h_c_5 = self.constraint_5(x, total_n)

        # # Combine all the constraints and objective function     
        h_final = penalty_a * (h_c_1 + h_c_2 + h_c_3 + h_c_4 + h_c_5) + \
            penalty_b * h_distance
        
        # # Create a mapping from variable labels to linear indices
        mapping = {}
        for i in range(total_n):
            for j in range(total_n):
                for r in range(self.R):
                    if i != j:
                        mapping[(i, j, r)] =  self.encode_var(i, j, r, total_n)


        # # Relabel the variables in the BQM
        linear_bqm = h_final.relabel_variables(mapping, inplace=False)

        return linear_bqm, penalty_b

    def generate_variables(self, total_n: int) -> dict[(int, int, int), \
                                                            dimod.Binary]:
        '''
        Creates a dictionary of variables x_i_j_r.

        Parameters
        ----------
        total_n : int
            The number of nodes in the TSP.

        Returns
        -------
        x : dict[(int, int, int), dimod.Binary]
            The variables of the QUBO.
        '''

        # # Creates a dictionary of variables x_i_j_r
        x = {(i, j, r): dimod.Binary((i, j, r)) for i in range(total_n) \
            for j in range(total_n) \
                for r in range(self.R)}
        
        # # Remove all elements where i == j
        for i in range(total_n):
            for j in range(total_n):
                if i == j:
                    for r in range(self.R):
                        del x[(i, j, r)]

        return x

    def distance_objective(self, tsp_matrix: np.ndarray, \
                        x: dict[(int, int, int), dimod.Binary], \
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
        for i in range(total_n):
            for j in range(total_n):
                if i != j:
                    if tsp_matrix[i][j] != 0:
                        h_distance += tsp_matrix[i][j] * x[(i, j, 1)]
                    else:
                        h_distance += (tsp_matrix.sum() ** 2) * x[(i, j, 1)]
        return h_distance

    def constraint_1(self, x: dict[(int, int, int), dimod.Binary], \
                     total_n: int) -> dimod.BinaryQuadraticModel:
        '''
        Constraint so that for each i, j only 1 r value must be given.

        Parameters
        ----------
        x : dict[(int, int, int), dimod.Binary]
            The variables of the QUBO.
        total_n : int
            The number of nodes in the TSP.

        Returns
        -------
        h_c_1 : dimod.BinaryQuadraticModel
            Constraint 1 of the QUBO.
        '''

        h_c_1 = 0
        for i in range(total_n):
            for j in range(total_n):
                temp = 0
                if i != j:
                    for r in range(self.R):
                        temp += x[(i, j, r)]
                    h_c_1 += (1 - temp) ** 2

        return h_c_1

    def constraint_2(self, x: dict[(int, int, int), dimod.Binary], total_n: int) -> \
        dimod.BinaryQuadraticModel:
        '''
        Constraint so that each node must be exited once.

        Parameters
        ----------
        x : dict[(int, int, int), dimod.Binary]
            The variables of the QUBO.
        total_n : int
            The number of nodes in the TSP.
        
        Returns
        -------
        h_c_2 : dimod.BinaryQuadraticModel
            Constraint 2 of the QUBO.
        '''

        h_c_2 = 0
        for i in range(total_n):
            temp = 0
            for j in range(total_n):
                if i != j:
                    temp += x[(i, j, 1)]
            h_c_2 += (1 - temp) ** 2
        return h_c_2

    def constraint_3(self, x: dict[(int, int, int), dimod.Binary], total_n: int) -> \
        dimod.BinaryQuadraticModel:
        '''
        Constraint so that each node must be reached once.

        Parameters
        ----------
        x : dict[(int, int, int), dimod.Binary]
            The variables of the QUBO.
        total_n : int
            The number of nodes in the TSP.

        Returns
        -------
        h_c_3 : dimod.BinaryQuadraticModel
            Constraint 3 of the QUBO.
        '''
        h_c_3 = 0
        for j in range(total_n):
            temp = 0
            for i in range(total_n):
                if i != j:
                    temp += x[(i, j, 1)]
            h_c_3 += (1 - temp) ** 2
        return h_c_3

    def constraint_4(self, x: dict[(int, int, int), dimod.Binary], total_n: int) -> \
        dimod.BinaryQuadraticModel:
        '''
        Constraint so that if node i is reached before j, then node j is reached 
        after i.

        Parameters
        ----------
        x : dict[(int, int, int), dimod.Binary]
            The variables of the QUBO.
        total_n : int
            The number of nodes in the TSP.

        Returns
        -------
        h_c_4 : dimod.BinaryQuadraticModel
            Constraint 4 of the QUBO.
        '''
        h_c_4 = 0
        for i in range(1, total_n):
            for j in range(i, total_n):
                if i != j:
                    h_c_4 += (1 - x[(i, j, 2)] - x[(j, i, 2)]) ** 2

        return h_c_4

    def constraint_5(self, x: dict[(int, int, int), dimod.Binary], total_n: int) -> \
        dimod.BinaryQuadraticModel:
        '''
        Constraint to prevent subtours and cycles.

        Parameters
        ----------
        x : dict[(int, int, int), dimod.Binary]
            The variables of the QUBO.
        total_n : int
            The number of nodes in the TSP.

        Returns
        -------
        h_c_5 : dimod.BinaryQuadraticModel
            Constraint 5 of the QUBO.
        '''

        h_c_5 = 0
        for i in range(1, total_n):
            for j in range(1, total_n):
                for k in range(1, total_n):
                    if i != j and i != k and j != k:
                        h_c_5 += x[(j, i, 2)] * x[(k, j, 2)] - \
                            x[(j, i, 2)] * x[(k, i, 2)] - \
                            x[(k, j, 2)] * x[(k, i, 2)] + \
                            x[(k, i, 2)]

        return h_c_5

    def find_invalid_constraints(self, solution, total_n) -> \
        tuple[list, dict, dict, list, list]:
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
            List of edges that break constraint 1.
        c_2 : dict
            Dictionary of nodes that break constraint 2.
            Where the key is the node, and the value is a list nodes the key node 
            goes to.
        c_3 : dict
            Dictionary of nodes that break constraint 3. 
            Where the key is the node, and the value is a list nodes the key node 
            is visited from.
        c_4 : list
            List of edges that break constraint 4.
        c_5 : list
            List of (i, j, k) values that break constraint 5.
        '''
        solution = self.decode_solution(solution, total_n)
        #  # Check that a for the same i and j only one r value is given
        c_1 = []
        for i in range(total_n):
            for j in range(total_n):
                temp = 0
                if i != j:
                    for r in range(self.R):
                            temp += solution[(i, j, r)]
                    if temp != 1:
                        c_1.append((i, j))

        # # Check that each node is exited once
        c_2 = {}
        for i in range(total_n):
            temp = 0
            exits = []
            for j in range(total_n):
                if i != j:
                    temp += solution[(i, j, 1)]
                    if solution[(i, j, 1)] == 1:
                        exits.append(j)
            if temp != 1:
                c_2[(i)] = exits 

        # # Check that each node is reached once
        c_3 = {}
        for j in range(total_n):
            temp = 0
            reached = []
            for i in range(total_n):
                if i != j:
                    temp += solution[(i, j, 1)]
                    if solution[(i, j, 1)] == 1:
                        reached.append(i)
            if temp != 1:
                c_3[(j)] = None if reached == [] else reached
        
        # # Check that if node i is reached before j, then node j is reached after i
        c_4 = []
        for i in range(1, total_n):
            for j in range(i, total_n):
                if i != j:
                    if solution[(i, j, 2)] == 0 and solution[(j, i, 2)] == 0:
                        c_4.append((i, j))
                    elif solution[(i, j, 2)] == 1 and solution[(j, i, 2)] == 1:
                        c_4.append((i, j))

        # # Check that no subtours or subcycles are created
        c_5 = []
        for i in range(1, total_n):
            for j in range(1, total_n):
                for k in range(1, total_n):
                    if i != j and i != k and j != k:
                        if solution[(j, i, 2)] == 1 and solution[(k, j, 2)] == 1 \
                            and solution[(k, i, 2)] == 0:
                            c_5.append((i, j, k))
                        elif solution[(j, i, 2)] == 0 and solution[(k, j, 2)] == 0 \
                            and solution[(k, i, 2)] == 1:
                            c_5.append((i, j, k))

        return c_1, c_2, c_3, c_4, c_5

    def validity_score(self, solution: dict, total_n: int) -> tuple[dict, list]:
        '''
        Calculate the validity score of a solution and a list of which constraints 
        are broken.

        Parameters
        ----------
        solution : dict
            The solution to the QUBO
        total_n : int
            The total number of nodes
        
        Returns
        -------
        int
            The number of broken constraints
        list
            The list of broken constraints
        '''
        # # Uses the find invalid constraints function to determine a score of how 
        # # many constraints are broken.
        constraints = self.find_invalid_constraints(solution, total_n)
        score = 0
        broken_constraints = []
        for i, constraint in enumerate(constraints):
            if len(constraint) > 0:
                score += 1
                broken_constraints.append(i + 1)

        return score, broken_constraints

    def convert_solution_to_edge_list(self, solution: dict, total_n: int) \
        -> list:
        '''
        Convert the solution to a list of edges

        Parameters
        ----------
        solution : dict
            The solution to be converted
        total_n : int
            The total number of nodes
        
        Returns
        -------
        list
            The edges in the solution
        '''
        edge_list = []
        decoded_solution = self.decode_solution(solution, total_n)

        for (i, j, k), value in decoded_solution.items():
            if k == 1 and value == 1:
                edge_list.append((i, j))
                
        return edge_list


def main():
    '''
    Main function
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    '''
    gps_qubo = GPSBQMFormulator()

    # fname = "TSP_Instances/tsp_5_2"
    fname = "TSP_Instances/five_d"
    tsp_matrix = np.loadtxt(fname)
    # tsp_matrix = gps_qubo.generate_tsp_matrix(6, 1)
    total_n = len(tsp_matrix)

    bqm, penalty_b = gps_qubo.form_bqm(tsp_matrix)
    # print(bqm)

    best_solution = gps_qubo.solve_bqm(bqm)
    encoded_solution = {}
    for i, value in enumerate(best_solution.items()):
        encoded_solution[(i)] = value[1]
    decoded_solution = gps_qubo.decode_solution(encoded_solution, total_n) 
    print('Decoding Works: ', decoded_solution==best_solution)
    
    print('Best Energy: ', bqm.energy(best_solution)/penalty_b)

    # # Get the invalid constraints
    # c_1, c_2, c_3, c_4, c_5 = find_invalid_constraints(best_solution, total_n)
    # print('c_1: ', c_1)
    # print('c_2: ', c_2)
    # print('c_3: ', c_3)
    # print('c_4: ', c_4)
    # print('c_5: ', c_5)

    # # Calculate a score relating to how many constraints are broken in the 
    # # solution. e.g. 0 means no constraints are broken, 5 means all 
    # # constraints are broken.
    # # Also list which constraints are broken
    score, broken_constraints = gps_qubo.validity_score(best_solution, total_n)
    print('Validity Score: ', score)
    print('Broken constraints: ', broken_constraints)

    route = gps_qubo.get_tour(best_solution, total_n)
    print('Route: ', route)
    cost = gps_qubo.tour_cost(best_solution, tsp_matrix)
    print('Cost: ', cost)

    gps_qubo.graph_solution(best_solution, tsp_matrix)

if __name__ == '__main__':
    main()
