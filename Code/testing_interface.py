import dwave.samplers as ds
import gps_tsp_bqm as gps
import native_tsp_bqm as native
import qaoa_solver as qaoa
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp
from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Tsp
from docplex.mp.model import Model
import logging
import sys
import dimod
import numpy as np
from datetime import datetime
import pandas as pd
import os

def test_qubo_implementation_sa():
    print('Test QUBO Implementation SA')
    print('------------------------')

    # # Load TSP Instances
    instance_names = ['tsp_3_0', 'tsp_4_0', 'five_d']
    tsp_instances = {}
    for name in instance_names:
        tsp_instances[name] = np.loadtxt('TSP_Instances/' + name)

    # # Create BQM
    native_bqm_formulator = native.NativeBQMFormulator()
    gps_bqm_formulator = gps.GPSBQMFormulator()
    bqm_formulators = [native_bqm_formulator, gps_bqm_formulator]
    
    sampler = ds.SimulatedAnnealingSampler()
    sampler.parameters = {'num_reads': 10, 'num_sweeps': 10000}

    df = pd.DataFrame(columns=['Model', 'Instance', 'Time(ms)', 'Cost', 'Energy'])
    count = 0
    for instance in tsp_instances.items():
        for bqm_formulator in bqm_formulators:
            for _ in range(10):
                count += 1
                # print('Instance: ', instance)
                # print('BQM Formulator: ', bqm_formulator)
                bqm, weight_parameter = bqm_formulator.form_bqm(instance[1])
                start_time = datetime.now()
                best_solution = bqm_formulator.solve_bqm(bqm, sampler)
                end_time = datetime.now()
                time = end_time - start_time
                energy = bqm.energy(best_solution)/weight_parameter
                cost = bqm_formulator.tour_cost(best_solution, instance[1])
                # print('Best Solution: ', best_solution)
                # print('Best Solution Energy: ', energy)
                # print('Cost: ', cost)
                model = (bqm_formulator.__class__.__name__).replace('BQMFormulator', '')
                if model == 'Native':
                    validity, _ = bqm_formulator.validity_score(best_solution, len(instance[1]), instance[1])
                else:
                    validity, _ = bqm_formulator.validity_score(best_solution, len(instance[1]))
                df_new = pd.DataFrame({'Model': model, 
                                       'Instance': instance[0], 
                                       'Time(ms)': time, 
                                       'Cost': cost, 
                                       'Energy': energy,
                                       'Validity': validity}, index=[count])
                df = pd.concat([df, df_new])

    # df['Time'] = pd.to_datetime(df['Time'])
    df['Time(ms)'] = df['Time(ms)'].dt.total_seconds() * 1000


    path = './Experiments/QUBO_Formulation_Test_SA'
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_csv(path + '/test_full.csv')
    print(df.to_string())


    avg_df = df.groupby(['Model', 'Instance']).mean()
    avg_df.to_csv(path + '/test_average.csv')
    print(avg_df.to_string())

    avg_df.to_latex(path + '/test_average.tex')
    print(avg_df.to_latex())

def test_qaoa_parameters():
    print('Test QAOA Parameters')
    print('------------------------')

    # # Load TSP Instances
    instance_names = ['tsp_3_0', 'tsp_4_0']
    tsp_instances = {}
    for name in instance_names:
        tsp_instances[name] = np.loadtxt('TSP_Instances/' + name)

    # # Create BQM
    native_bqm_formulator = native.NativeBQMFormulator()
    gps_bqm_formulator = gps.GPSBQMFormulator()
    bqm_formulators = [native_bqm_formulator, gps_bqm_formulator]
    
    sampler = qaoa.QiskitQAOASampler()
    
    for instance in tsp_instances.items():
        for bqm_formulator in bqm_formulators:
            for j in range(2):
                for i in range(3):
                    shots = 1000 * (10 ** j)

                    print('Instance: ', instance[0])
                    print('Iteration: ', i)
                    bqm, weight_parameter = bqm_formulator.form_bqm(instance[1])
                    model = (bqm_formulator.__class__.__name__).replace('BQMFormulator', '')
                    print('Model: ', model)
                    fname = './Experiments/QAOA_Parameters/' + model + '_' + \
                        instance[0] + '_' + str(shots) + '_' + str(i) + '.pdf'

                    energies, beta_max, gamma_max = sampler.grid_search(bqm, shots, 10)

                    plt.ylabel(r"$\gamma$")
                    plt.xlabel(r"$\beta$")
                    plt.title("Energy as a function of parameters")
                    plt.imshow(energies, extent=(0, beta_max, gamma_max, 0))
                    plt.colorbar()
                    # plt.show()
                    plt.savefig(fname)
                    plt.clf()
    

def test_qaoa_p():
    print('Test QAOA Implementation P')
    print('------------------------')

    # # Load TSP Instances
    instance_names = ['tsp_3_0']
    tsp_instances = {}
    for name in instance_names:
        tsp_instances[name] = np.loadtxt('TSP_Instances/' + name)

    # # Create BQM
    native_bqm_formulator = native.NativeBQMFormulator()
    gps_bqm_formulator = gps.GPSBQMFormulator()
    bqm_formulators = [native_bqm_formulator, gps_bqm_formulator]
    
    sampler = qaoa.QiskitQAOASampler(maxiter=100, n_shots=1000, method='Nelder-Mead')

    df = pd.DataFrame(columns=['Model', 'Instance', 'p', 'Time(ms)', 'Cost', 'Energy'])
    count = 0
    for instance in tsp_instances.items():
        for bqm_formulator in bqm_formulators:
            for p in [3, 11]:
                sampler.p = p
                count += 1
                # print('Instance: ', instance)
                # print('BQM Formulator: ', bqm_formulator)
                bqm, weight_parameter = bqm_formulator.form_bqm(instance[1])
                start_time = datetime.now()
                best_solution = bqm_formulator.solve_bqm(bqm, sampler)
                end_time = datetime.now()
                time = end_time - start_time
                energy = bqm.energy(best_solution)/weight_parameter
                cost = bqm_formulator.tour_cost(best_solution, instance[1])
                # print('Best Solution: ', best_solution)
                # print('Best Solution Energy: ', energy)
                # print('Cost: ', cost)
                model = (bqm_formulator.__class__.__name__).replace('BQMFormulator', '')
                if model == 'Native':
                    validity, _ = bqm_formulator.validity_score(best_solution, len(instance[1]), instance[1])
                else:
                    validity, _ = bqm_formulator.validity_score(best_solution, len(instance[1]))
                df_new = pd.DataFrame({'Model': model, 
                                       'Instance': instance[0], 
                                       'p': p,
                                       'Time(ms)': time, 
                                       'Cost': cost, 
                                       'Energy': energy,
                                       'Validity': validity}, index=[count])
                df = pd.concat([df, df_new])

    # df['Time'] = pd.to_datetime(df['Time'])
    df['Time(ms)'] = df['Time(ms)'].dt.total_seconds() * 1000


    path = './Experiments/QAOA_P_Test'
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_csv(path + '/test_full.csv')
    print(df.to_string())

    df.to_latex(path + '/test_average.tex')
    print(df.to_latex())

def test_qaoa_5n():
    print('Test QAOA Implementation 5N')
    print('------------------------')

    # # Load TSP Instances
    instance_names = ['five_d']
    tsp_instances = {}
    for name in instance_names:
        tsp_instances[name] = np.loadtxt('TSP_Instances/' + name)

    # # Create BQM
    native_bqm_formulator = native.NativeBQMFormulator()
    bqm_formulators = [native_bqm_formulator]
    
    sampler = qaoa.QiskitQAOASampler(maxiter=100, n_shots=1000, method='Nelder-Mead')

    df = pd.DataFrame(columns=['Model', 'Instance', 'p', 'Time(ms)', 'Cost', 'Energy'])
    count = 0
    for instance in tsp_instances.items():
        for bqm_formulator in bqm_formulators:
            for p in [3]:
                sampler.p = p
                count += 1
                # print('Instance: ', instance)
                # print('BQM Formulator: ', bqm_formulator)
                bqm, weight_parameter = bqm_formulator.form_bqm(instance[1])
                start_time = datetime.now()
                best_solution = bqm_formulator.solve_bqm(bqm, sampler)
                end_time = datetime.now()
                time = end_time - start_time
                energy = bqm.energy(best_solution)/weight_parameter
                cost = bqm_formulator.tour_cost(best_solution, instance[1])
                # print('Best Solution: ', best_solution)
                # print('Best Solution Energy: ', energy)
                # print('Cost: ', cost)
                model = (bqm_formulator.__class__.__name__).replace('BQMFormulator', '')
                if model == 'Native':
                    validity, _ = bqm_formulator.validity_score(best_solution, len(instance[1]), instance[1])
                else:
                    validity, _ = bqm_formulator.validity_score(best_solution, len(instance[1]))
                df_new = pd.DataFrame({'Model': model, 
                                       'Instance': instance[0], 
                                       'p': p,
                                       'Time(ms)': time, 
                                       'Cost': cost, 
                                       'Energy': energy,
                                       'Validity': validity}, index=[count])
                df = pd.concat([df, df_new])

    # df['Time'] = pd.to_datetime(df['Time'])
    df['Time(ms)'] = df['Time(ms)'].dt.total_seconds() * 1000


    path = './Experiments/QAOA_5N_Test'
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_csv(path + '/test_full.csv')
    print(df.to_string())

    df.to_latex(path + '/test_average.tex')
    print(df.to_latex())



def sams_qaoa_test():
    print('Test My Implementation')
    print('----------')
    # filename = 'native_instances/native_bqm_5_0'
    # bqm_formulator = native.NativeBQMFormulator()
    # bqm, weight_factor = bqm_formulator.load_bqm(filename)
    # bqm_formulator = native.NativeBQMFormulator()
    bqm_formulator = gps.GPSBQMFormulator()
    # tsp_matrix = bqm_formulator.generate_tsp_matrix(3, 1)
    # fname = "TSP_Instances/tsp_5_2"
    # fname = "TSP_Instances/five_d"
    fname = "TSP_Instances/tsp_3_0"
    tsp_matrix = np.loadtxt(fname)
    total_n = len(tsp_matrix)
    bqm, weight_factor = bqm_formulator.form_bqm(tsp_matrix)


    sampler = qaoa.QiskitQAOASampler(backend_name='qasm_simulator', p=3, \
                                     maxiter=100, n_shots=1000, \
                                        method='Nelder-Mead', output=True)

    # # Output Circuit
    # p = 1
    # initial_params = np.random.uniform(0, 2 * np.pi, 2 * p)
    # Q = sampler.bqm_to_q_matrix(bqm)
    # # print('Q: ', Q)
    # # print('BQM: ', bqm)
    # circuit = sampler.qaoa_circuit(Q, initial_params)
    # # print(circuit)
    # circuit.draw(filename='qaoa_circuit.png', output='mpl')

    print('Solving...')

    # # Set up logging
    # logging.basicConfig(filename='qaoa_mine.log', filemode='w', level=logging.INFO)
    # logger = logging.getLogger(qaoa.__class__.__name__)
    # # Run the QAOA
    # console_output = sys.stdout
    # sys.stdout = open('test_Native_5_2.txt', 'w')
    # print('Solving...')
    # best_solution = bqm_formulator.solve_bqm(bqm, sampler)
    sample = sampler.sample(bqm)
    print('Sample: ', sample.record['metadata'])
    best_solution = sample.first.sample
    # logger.setLevel(logging.DEBUG)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)
    # logger.addHandler(console_handler)

    print('Best Solution: ', best_solution)
    print('Best Solution Energy: ', \
                bqm.energy(best_solution)/weight_factor)
    edge_list = bqm_formulator.convert_solution_to_edge_list(best_solution, total_n)
    print('Edge List: ', edge_list)
    route = bqm_formulator.get_tour(best_solution, len(tsp_matrix))
    cost = bqm_formulator.tour_cost(best_solution, tsp_matrix)
    print('Route: ', route)
    print('Cost: ', cost)
    # Graph the solution
    bqm_formulator.graph_solution(best_solution, tsp_matrix)

    # sys.stdout.close()
    # sys.stdout = console_output


def qiskit_test():
    print('Test Qiskit Implementation')
    print('--------------------------')

    # # Create BQM
    bqm_formulator = native.NativeBQMFormulator()
    tsp_matrix = bqm_formulator.generate_tsp_matrix(4, 1)
    total_n = len(tsp_matrix)
    bqm, weight_factor = bqm_formulator.form_bqm(tsp_matrix)


    # Convert the BQM to a QuadraticProgram
    bqm_spin = bqm.change_vartype(vartype='SPIN', inplace=False)
    qp = bqm_formulator.convert_bqm_to_quadratic_program(bqm_spin)
    print(qp.prettyprint())
    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)
    qubitOp, offset = qubo.to_ising()
    # print("Offset:", offset)
    # print("Ising Hamiltonian:")
    # print(str(qubitOp))

    # # Set up logging
    logging.basicConfig(filename='qaoa_qiskit.log', filemode='w', level=logging.INFO)

    # # # Set up the Exact Solver
    # exact_eigensolver = NumPyMinimumEigensolver()
    # result  = exact_eigensolver.compute_minimum_eigenvalue(qubitOp)

    # # # Set up QAOA
    optimizer = SPSA(maxiter=100)
    qaoa = QAOA(sampler=Sampler(), optimizer=optimizer, reps=4)


    # # root = logging.getLogger()
    # # root.setLevel(logging.DEBUG)

    # # # # handler = logging.StreamHandler(sys.stdout)
    # # # # handler.setLevel(logging.DEBUG)

    # # Run the QAOA
    result = qaoa.compute_minimum_eigenvalue(qubitOp)

    # # logger = logging.getLogger(qaoa.__class__.__name__)
    # # logger.setLevel(logging.DEBUG)
    # # console_handler = logging.StreamHandler()
    # # console_handler.setLevel(logging.DEBUG)
    # # logger.addHandler(console_handler)
    # # # ry = TwoLocal(qubitOp.num_qubits, "ry", "cz", reps=5, entanglement="linear")
    # # # vqe = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=optimizer)

    # # # result = vqe.compute_minimum_eigenvalue(qubitOp)

    # # Output the results
    print("energy:", result.eigenvalue.real)
    print("tsp objective:", result.eigenvalue.real + offset)
    # print("time:", result.optimizer_time)
    n = 4
    tsp = Tsp.create_random_instance(n, seed=123)
    x = tsp.sample_most_likely(result.eigenstate)
    print("feasible:", qubo.is_feasible(x))
    print("solution:", x)
    solution_dict = {}
    for i in range(len(x)):
        solution_dict[i] = x[i]
    route = bqm_formulator.get_tour(solution_dict, total_n)
    print('Route: ', route)
    cost = bqm_formulator.tour_cost(solution_dict, tsp_matrix)
    print('Cost: ', cost)
    # Graph the solution
    bqm_formulator.graph_solution(solution_dict, tsp_matrix)

def qiskit_test2():
    total_n = 5
    bqm_formulator = native.NativeBQMFormulator()
    tsp_matrix = bqm_formulator.generate_tsp_matrix(total_n, 1)

    bqm, weight_factor = bqm_formulator.form_bqm(tsp_matrix)

    # # Get the number of variables in the BQM
    # num_vars = len(bqm.variables)

    # # Initialize a matrix of zeros to represent the Q matrix
    # Q = np.zeros((num_vars, num_vars))

    # # Fill in the non-zero entries of the Q matrix from the BQM
    # for (i, j), val in bqm.quadratic.items():
    #     Q[i, j] = val
    #     Q[j, i] = val

    # # Add the linear biases to the diagonal of the Q matrix
    # for i, val in bqm.linear.items():
    #     Q[i, i] = val

    # solution, energy = solve(Q)
    # print('Solution: ', solution)
    # print('Energy: ', energy)
    
    # best_solution = {}
    # for i in range(len(solution[0])):
    #     best_solution[i] = solution[0][i]

    mdl = create_docplex_model(tsp_matrix)
    mod = from_docplex_mp(mdl)
    print(mod.prettyprint())

    def print_iteration_num(num_iter, param, mean, std):
        time = datetime.utcnow()
        print(f"Iteration {num_iter}: energy = {mean} {time}")

    optimizer = COBYLA(maxiter=100)
    qaoa = QAOA(sampler=Sampler(), optimizer=optimizer, reps=11, callback=print_iteration_num)

    qaoa = MinimumEigenOptimizer(qaoa)
    result = qaoa.solve(mod)

    # exact_eigensolver = NumPyMinimumEigensolver()
    # solver = MinimumEigenOptimizer(exact_eigensolver)
    # result = solver.solve(mod)
    print(result)
    solution = result.x
    print(solution)

    edge_list = []
    count = 0
    for i in range(total_n):
        for j in range(total_n):
            if i != j:
                if solution[count] == 1:
                    edge_list.append((i, j))
                count += 1
    print(edge_list)


    # tour = bqm_formulator.get_tour(best_solution, total_n)
    # cost = bqm_formulator.tour_cost(best_solution, tsp_matrix)
    # print('Tour: ', tour)
    # print('Cost: ', cost)
    # bqm_formulator.graph_solution(best_solution, tsp_matrix)

def create_docplex_model(distances):
    # Create a Docplex model
    mdl = Model(name='tsp')

    num_cities = len(distances)

    # Add binary variables
    x = {}
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                x[(i, j)] = mdl.binary_var(name='x_{0}_{1}'.format(i, j))

    # Add objective function
    mdl.minimize(mdl.sum(distances[i][j]*x[(i, j)] for i in range(num_cities) for j in range(num_cities) if i != j))

    # Add constraints
    for i in range(num_cities):
        mdl.add_constraint(mdl.sum(x[(i, j)] for j in range(num_cities) if i != j) == 1)
    for j in range(num_cities):
        mdl.add_constraint(mdl.sum(x[(i, j)] for i in range(num_cities) if i != j) == 1)
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                mdl.add_constraint(x[(i, j)] + x[(j, i)] <= 1)

    return mdl

def grid_search_test():
    bqm_formulator = native.NativeBQMFormulator()
    tsp_matrix = bqm_formulator.generate_tsp_matrix(4, 1)
    # fname = "TSP_Instances/tsp_5_2"
    # tsp_matrix = np.loadtxt(fname)
    total_n = len(tsp_matrix)

    bqm, weight_factor = bqm_formulator.form_bqm(tsp_matrix)


    sampler = qaoa.QiskitQAOASampler(backend_name='qasm_simulator', p=1, \
                                     maxiter=100, n_shots=10000, output=False)

    print('Solving...')

    grid_size = 50
    sampler.grid_search(bqm, grid_size)


def solve(Q):
    print('Solving...')
    qp = QuadraticProgram()
    [qp.binary_var() for _ in range(Q.shape[0])]
    qp.minimize(quadratic=Q)

    # # # Set up QAOA
    # optimizer = SPSA(maxiter=100)
    # optimizer = COBYLA()
    # qaoa = QAOA(sampler=Sampler(), optimizer=optimizer, reps=20)
    # # Set up the Exact Solver
    exact_eigensolver = NumPyMinimumEigensolver()
    solver = MinimumEigenOptimizer(exact_eigensolver)
    result = solver.solve(qp)
    # result  = exact_eigensolver.compute_minimum_eigenvalue(qubitOp)
    # qaoa = MinimumEigenOptimizer(qaoa)
    # qaoa_result = qaoa.solve(qp)
    # return [qaoa_result.x], [qaoa_result.fval]
    return [result.x], [result.fval]

if __name__ == "__main__":
    # sams_qaoa_test()

    # test_qubo_implementation_sa()

    # test_qaoa_parameters()
    
    # test_qaoa_p()
    test_qaoa_5n()

    # grid_search_test()

    # qiskit_test()