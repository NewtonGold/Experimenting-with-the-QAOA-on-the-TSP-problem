import dimod
import numpy as np
from qiskit import QuantumCircuit, transpile, Aer, execute
from scipy.optimize import minimize
from datetime import datetime
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
import matplotlib.pyplot as plt


class QiskitQAOASampler(dimod.Sampler):

    def __init__(self, backend_name: str = 'qasm_simulator', p: int = 1,
                 maxiter: int = 1000, n_shots: int = 1024,
                 method: str = 'COBYLA', output: bool = False) -> None:
        '''
        Initialize the Qiskit QAOA sampler

        Parameters
        ----------
        backend_name : str, optional
            The name of the backend to be used, by default 'qasm_simulator'
        p : int, optional
            The number of layers of the QAOA circuit, by default 1
        maxiter : int, optional
            The maximum number of iterations for the optimizer, by default 1000
        n_shots : int, optional
            The number of shots to be used, by default 1024
        method : str, optional
            The optimization method to be used, by default 'COBYLA'
        output : bool, optional
            Whether to print part of the QAOA process, by default False

        Returns
        -------
        None

        '''

        self.p = p
        self.maxiter = maxiter
        self.n_shots = n_shots
        self.num_iter = 0
        self.method = method
        self.output = output

        self._properties = {'description': 'a sampler for the Quantum ' +
                            'Approximate Optimization Algorithm (QAOA)'}
        self._parameters = {'verbose': []}

        if backend_name == 'qasm_simulator':
            self.backend = Aer.get_backend(backend_name)
        else:
            # # Load your IBMQ account (replace 'YOUR_API_KEY' with your
            # # actual API key)
            # # IBMQ.save_account('YOUR_API_KEY')
            IBMQ.load_account()
            if backend_name == 'least_busy':
                provider = IBMQ.get_provider(hub='ibm-q')
                backends = provider.backends(filters=lambda b:
                                             b.status().operational and
                                             not b.configuration().simulator)
                self.backend = least_busy(backends)
            else:
                provider = IBMQ.get_provider(hub='ibm-q')
                self.backend = provider.get_backend(backend_name)

    def qaoa_circuit(self, bqm: dimod.BinaryQuadraticModel,
                     params: np.ndarray, p: int) -> QuantumCircuit:
        '''
        Construct the QAOA circuit

        Parameters
        ----------
        bqm : dimod.BinaryQuadraticModel
            The binary quadratic model to be solved in spin form
        params : np.ndarray
            The parameters of the QAOA circuit
        p : int
            The number of layers of the QAOA circuit

        Returns
        -------
        QuantumCircuit
            The QAOA circuit


        Notes
        -----
        The QAOA circuit is constructed as follows:
        1. Prepare the initial equal superposition state
        2. Apply the alternating layers of U_B and U_P
        3. Measure all qubits

            '''
        n_qubits = len(bqm)
        betas, gammas = params[:p], params[p:]

        # # Prepare the initial equal superposition state
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))

        # # Apply the alternating layers of U_C and U_B
        for layer in range(p):
            # # Apply U_B
            qc.rx(2 * betas[layer], range(n_qubits))

            # # Apply U_P
            for (i, j), coeff in bqm.quadratic.items():
                qc.cx(i, j)
                qc.rz(2 * gammas[layer] * coeff, j)
                qc.cx(i, j)

            for i, coeff in bqm.linear.items():
                qc.rz(2 * gammas[layer] * coeff, i)

        qc.measure_all()
        return qc

    def qaoa_objective_function(self, params: np.ndarray,
                                bqm_spin: dimod.BinaryQuadraticModel, p: int) \
            -> float:
        '''
        The objective function of the QAOA circuit

        Parameters
        ----------
        params : np.ndarray
            The parameters of the QAOA circuit
        bqm_spin : dimod.BinaryQuadraticModel
            The binary quadratic model to be solved in spin format
        p : int
            The number of layers of the QAOA circuit

        Returns
        -------
        float
            The average energy of the QAOA circuit

        '''
        optimal_circuit = self.qaoa_circuit(bqm_spin, params, p)
        transpiled_circuit = transpile(optimal_circuit, self.backend)
        job = execute(transpiled_circuit, self.backend, shots=self.n_shots)
        counts = job.result().get_counts()

        energy = 0
        for bitstring, count in counts.items():
            # # Convert binary variables to spin variables
            s = np.array([2 * int(bit) - 1 for bit in bitstring])
            # # Calculate energy using the binary BQM
            energy += count * bqm_spin.energy(s)

        energy /= self.n_shots
        if self.output:
            self.print_iteration_num(energy)
        return energy

    def print_iteration_num(self, mean: float) -> None:
        '''
        Print the iteration number and the average energy at each cycle of the
        QAOA circuit

        Parameters
        ----------
        mean : float
            The average energy of the QAOA circuit

        Returns
        -------
        None        
        '''
        time = datetime.utcnow()
        print(f"Iteration {self.num_iter}: energy = {mean} {time}")
        self.num_iter += 1

    def qaoa_bqm_solver(self, bqm: dimod.BinaryQuadraticModel, p: int = 1,
                        maxiter: int = 1000, n_shots: int = 1024,
                        method: str = 'COBYLA') -> dict[(QuantumCircuit,
                                                         np.ndarray, str,
                                                         float)]:
        '''
        Solve the BQM problem using the QAOA circuit

        Parameters
        ----------
        bqm : dimod.BinaryQuadraticModel
            The binary quadratic model to be solved in binary form
        p : int, optional
            The number of layers of the QAOA circuit, by default 1
        maxiter : int, optional
            The maximum number of iterations of the classical optimizer,
            by default 1000
        n_shots : int, optional
            The number of shots of the quantum circuit, by default 1024
        method : str, optional
            The classical optimizer used to optimize the QAOA circuit,
            by default 'COBYLA'

        Returns
        -------
        dict
            The optimal circuit, optimal parameters, optimal bitstring and the 
            corresponding energy.

        '''
        # # Convert BQM with binary variables to BQM with spin variables
        bqm_spin = bqm.change_vartype(dimod.SPIN, inplace=False)

        # # Initialize the parameters of the QAOA circuit
        initial_params = np.random.uniform(0, 2 * np.pi, 2 * p)
        self.num_iter = 0

        # # Optimize the QAOA circuit using the scipy minimize function
        result = minimize(self.qaoa_objective_function, initial_params,
                          args=(bqm_spin, p), method=method,
                          options={'maxfev': maxiter})

        # # Extract the optimal parameters
        optimal_params = result['x']
        print("Optimal parameters:", optimal_params)

        # # Create the optimal circuit
        optimal_circuit = self.qaoa_circuit(bqm_spin, optimal_params, p)
        transpiled_circuit = transpile(optimal_circuit, self.backend)

        # # Sample the optimal circuit
        job = execute(transpiled_circuit, self.backend, shots=n_shots)
        counts = job.result().get_counts()

        # # Find the most bitstring with the lowest energy
        best_energy = np.inf
        for bitstring, _ in counts.items():
            x = np.array([int(bit) for bit in bitstring])
            energy = bqm.energy(x)
            if energy < best_energy:
                best_energy = energy
                best_bitstring = bitstring

        # # Convert the best bitstring to binary variables
        # most_common_bitstring = max(counts, key=counts.get)
        optimal_solution = [int(bit) for bit in best_bitstring[::-1]]
        # # Calculate the energy of the optimal bitstring
        optimal_value = bqm.energy(optimal_solution)

        return {'circuit': optimal_circuit,
                'parameters': optimal_params,
                'solution': optimal_solution,
                'value': optimal_value,
                'num_iterations': result['nfev']+1}

    def sample(self, bqm: dimod.BinaryQuadraticModel) -> dimod.SampleSet:
        '''
        Sample from the QAOA circuit

        Parameters
        ----------
        bqm : dimod.BinaryQuadraticModel
            The binary quadratic model to be solved in binary form

        Returns
        -------
        dimod.SampleSet
            The optimal bitstring and the corresponding energy.

        '''
        result = self.qaoa_bqm_solver(bqm, p=self.p, maxiter=self.maxiter,
                                      n_shots=self.n_shots, method=self.method)
        metadata = {'num_iterations': result['num_iterations']}
        sample = dimod.SampleSet.from_samples([result['solution']],
                                              energy=result['value'],
                                              vartype=dimod.BINARY,
                                              metadata=metadata)
        # # result['circuit'].draw(output='mpl', filename='qaoa_circuit.png')
        return sample

    def grid_search(self, bqm: dimod.BinaryQuadraticModel, shots: int, \
                    grid_size: int = 10,) -> None:
        '''
        Do a grid search over the parameters of the QAOA circuit

        Parameters
        ----------
        bqm : dimod.BinaryQuadraticModel
            The binary quadratic model to be solved in binary form
        grid_size : int, optional
            The number of points in the grid search, by default 10
        fname : str
            The filename of the plot

        Returns
        -------
        None

        '''
        self.n_shots = shots
        gamma_max = 2 * np.pi
        beta_max = 2 * np.pi
        params = [0, 0]

        # # Convert BQM with binary variables to BQM with spin variables
        bqm_spin = bqm.change_vartype(dimod.SPIN, inplace=False)

        # Do the grid search.
        energies = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                params[0] = i * gamma_max / grid_size
                params[1] = j * beta_max / grid_size
                energies[i, j] = self.qaoa_objective_function(params, bqm_spin, 1)

        return energies, beta_max, gamma_max

    @property
    def properties(self):
        return self._properties

    @property
    def parameters(self):
        return self._properties


def main():
    # # Example usage
    Q = np.array([
        [1, -1, 0],
        [-1, 2, -1],
        [0, -1, 1]
    ])

    bqm = dimod.BinaryQuadraticModel(Q, "BINARY")
    sampler = QiskitQAOASampler(maxiter=100)
    response = sampler.sample(bqm)
    print("Optimal solution:", response.first.sample)
    print("Optimal value:", response.first.energy)


if __name__ == '__main__':
    main()
