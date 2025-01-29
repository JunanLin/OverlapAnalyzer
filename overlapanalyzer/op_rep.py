import numpy as np
from openfermion import (
    QubitOperator, 
    get_ground_state, 
    get_sparse_operator, 
    load_operator, 
    fermi_hubbard, 
    jordan_wigner
)
import os
import pickle

class SparseVector:
    def __init__(self):
        self.data = {}
    def add_entry(self, index, value):
        if isinstance(index, str):
            if not all(char in ['0', '1'] for char in index):
                raise ValueError("Invalid binary string entry")
            index = int(index, 2)
        self.data[index] = value

    def remove_entry(self, index):
        if index in self.data:
            del self.data[index]

    def remove_smallest_entries(self, num_entries_kept=None, threshold=None, aggressive=False):
        if num_entries_kept is not None and threshold is not None:
            if aggressive:
                num_entries_kept = min(num_entries_kept, len(self.data) - sum(value <= threshold for value in self.data.values()))
            else:
                num_entries_kept = max(num_entries_kept, len(self.data) - sum(value <= threshold for value in self.data.values()))
        elif num_entries_kept is not None:
            num_entries_kept = min(num_entries_kept, len(self.data))
        elif threshold is not None:
            num_entries_kept = len(self.data) - sum(value <= threshold for value in self.data.values())
        else:
            return

        if num_entries_kept >= len(self.data):
            return
        num_entries_to_remove = len(self.data) - num_entries_kept
        sorted_entries = sorted(self.data.items(), key=lambda x: x[1])
        for index, _ in sorted_entries[:num_entries_to_remove]:
            del self.data[index]

    def __add__(self, other):
        result = SparseVector()
        for index in set(self.data.keys()).union(other.data.keys()):
            result.data[index] = self.data.get(index, 0) + other.data.get(index, 0)
        return result

    def __sub__(self, other):
        result = SparseVector()
        for index in set(self.data.keys()).union(other.data.keys()):
            result.data[index] = self.data.get(index, 0) - other.data.get(index, 0)
        return result

    def __mul__(self, scalar):
        return self.scalar_multiply(scalar)

    def __rmul__(self, scalar):
        return self.scalar_multiply(scalar)

    def scalar_multiply(self, scalar):
        result = SparseVector()
        for index, value in self.data.items():
            result.data[index] = scalar * value
        return result

    def conjugate(self):
        result = SparseVector()
        for index, value in self.data.items():
            result.data[index] = value.conjugate()
        return result

    def dot(self, other):
        # Double-Check: do we need to conjugate other?
        result = 0
        for index in set(self.data.keys()).intersection(other.data.keys()):
            result += self.data[index] * other.data[index].conjugate()
        return result

    def norm(self):
        result = 0
        for value in self.data.values():
            result += abs(value) ** 2
        return result ** 0.5
    def assign_data(self, data_dict):
        self.data = data_dict.copy()

class SymplecticPauli():
    """
    Base class for symplectic representation of Pauli operators.
    A SymplecticPauli object can represent linear combinations of arbitrary Pauli operators.
    """
    def __init__(self):
        self.x_array = np.array([], dtype=int)
        self.z_array = np.array([], dtype=int)
        self.coeff_array = np.array([])
    def _add_term(self, x, z, coefficient):
        self.x_array = np.append(self.x_array, x)
        self.z_array = np.append(self.z_array, z)
        self.coeff_array = np.append(self.coeff_array, coefficient)
    def _apply_1_term_to_SparseVector(self, x, z, state):
        new_state = SparseVector()
        for index, value in state.data.items():
            # Apply X (bit flip)
            new_index = index ^ x
            
            # Apply Z (phase flip)
            phase = (-1) ** bin(index & z).count('1')
            
            new_value = phase * value
            
            if new_index in new_state.data:
                new_state.data[new_index] += new_value
            else:
                new_state.data[new_index] = new_value
    def apply_to_SparseVector(self, state:SparseVector):
        result = SparseVector()
        result.assign_data(state.data)
        for x, z, coeff in zip(self.x_array, self.z_array, self.coeff_array):
            term_result = self._apply_1_term_to_SparseVector(x, z, result)
            result += coeff * term_result
        return result

def term_to_symplectic(term):
    x = 0
    z = 0
    for qubit, operator in term:
        if operator == 'X':
            x |= 1 << qubit
        elif operator == 'Z':
            z |= 1 << qubit
        elif operator == 'Y':
            x |= 1 << qubit
            z |= 1 << qubit
    return x, z

def pauli_operator_to_symplectic(pauli_operator:QubitOperator):
    symplectic_pauli = SymplecticPauli()
    for term, coefficient in pauli_operator.terms.items():
        x, z = term_to_symplectic(term)
        # Convert coefficient to real if it's close to real
        if np.isclose(coefficient.imag, 0):
            coefficient = np.real(coefficient)
        symplectic_pauli._add_term(x, z, coefficient)
    return symplectic_pauli

def convert_PauliOperator_File_to_SymplecticPauli_File(file_name, file_path):
    qubit_operator = load_operator(file_name = file_name, data_directory=file_path, plain_text=True)
    symplectic =  pauli_operator_to_symplectic(qubit_operator)
    # Save with pickle
    with open(os.path.join(file_path, file_name[:-5] + '.pkl'), 'wb') as f:
        pickle.dump(symplectic, f)

def read_SymplecticPauli_file(file_path):
    # Use pickle to read the file
    with open(file_path, 'rb') as f:
        symplectic = pickle.load(f)
    return symplectic

def apply_Pauli_term_to_state(term, coefficient, state):
    current_state = state
    new_state = SparseVector()
    # if current_state is None:
    #     print('current_state is None')
    # else:
    #     print(f'current_state.data: {current_state.data}')
    for qubit, operator in term:
        # print(qubit, operator)
        if operator == 'X':
            # X|0> = |1>, X|1> = |0>
            flip_bit = 1 << qubit
            for index, value in current_state.data.items():
                new_index = index ^ flip_bit
                new_state.add_entry(new_index, value)
        elif operator == 'Y':
            # Y|0> = i|1>, Y|1> = -i|0>
            flip_bit = 1 << qubit
            for index, value in current_state.data.items():
                bit_was_1 = index & flip_bit != 0
                # If bit was 1, multiply by -i, else multiply by i
                new_index = index ^ flip_bit
                new_state.add_entry(new_index, -1j * value if bit_was_1 else 1j * value)
        elif operator == 'Z':
            # Z|0> = |0>, Z|1> = -|1>
            for index, value in current_state.data.items():
                bit_was_1 = index & (1 << qubit) != 0
                new_state.add_entry(index, -1 * value if bit_was_1 else value)
        current_state = new_state
        new_state = SparseVector()
    return coefficient * current_state 

def apply_Pauli_to_SparseVector(qubit_operator, state):
    new_state = SparseVector()
    for term, coefficient in qubit_operator.terms.items():
        new_state += apply_Pauli_term_to_state(term, coefficient, state)
    return new_state

def arnoldi(qubit_operator, initial_state, iterations):
    """
    Arnoldi method to find the lowest eigenvalue of a QubitOperator.
    """
    subspace_size = iterations
    Q = [None] * (iterations+1)
    H = np.zeros((iterations+1, iterations), dtype=np.complex128)
    with open('arnoldi_nonzero_elements.csv', 'a') as f:
        f.write(f'{0}, {len(initial_state.data)}\n')
    Q[0] = 1 / initial_state.norm() * initial_state
    for k in range(iterations):
        print(f'Iteration {k}: \n')
        v = apply_Pauli_to_SparseVector(qubit_operator, Q[k])
        print(f"Number of non-zero elements in v at iteration {k}: ", len(v.data))
        # Save the number of non-zero elements in v and the iteration number to a csv file
        with open('arnoldi_nonzero_elements.csv', 'a') as f:
            f.write(f'{k+1}, {len(v.data)}\n')

        for j in range(k+1):
            H[j, k] = Q[j].dot(v)
            v = v - Q[j] * H[j, k]
        H[k+1, k] = v.norm()
        # print(f'H: \n{H}')
        if H[k+1, k] != 0 and k != iterations-1:
            Q[k+1] = 1 / H[k+1, k] * v
        else:
            subspace_size = k
            # print(f'Truncated H: \n{H[:subspace_size, :subspace_size]}')
            break

    # Solve the subspace eigenvalue problem using the square part of H
    eigenvalues, _ = np.linalg.eig(H[:subspace_size, :subspace_size])
    return eigenvalues


# A test function for applying a QubitOperator to a SparseVector state
def test_apply_operator_to_state():
    state = SparseVector()
    state.add_entry(4, 1)
    state.add_entry(5, 1)
    state.add_entry(6, 1)
    state.add_entry(7, 1)
    qubit_operator = QubitOperator('X0 Y1 Z2', 1.0) + QubitOperator('Y3 Z4', 2.0)
    new_state = apply_Pauli_to_SparseVector(qubit_operator, state)
    print(new_state.data)
    # assert new_state.data == {7: -1j, 6: -1j, 5: 1j, 4: 1j}

def test_arnoldi():
    qubit_operator = QubitOperator('X0 Y1 Z2', 1.0) + QubitOperator('Y3 Z4', 2.0) + QubitOperator(' ', 20.0)
    initial_state = SparseVector()
    initial_state.add_entry(4, 1)
    initial_state.add_entry(5, 1)
    initial_state.add_entry(6, 1)
    initial_state.add_entry(7, 1)
    arnoldi_evals = arnoldi(qubit_operator, initial_state, 8)
    print("Ensuring that the eigenvalues are close to real: ", np.isclose(arnoldi_evals.imag, 0).all())
    # Cast the eigenvalues to real
    real_evals = arnoldi_evals.real
    lowest_eigenvalue = np.min(real_evals)
    print("Arnoldi ground state energy: ", lowest_eigenvalue)
    ground_state_energy, _ = get_ground_state(get_sparse_operator(qubit_operator))
    print("Ground state energy: ", ground_state_energy)
    assert np.isclose(lowest_eigenvalue, ground_state_energy)

def test_arnoldi_molecular_ham():
    from overlapanalyzer.read_ham import find_files
    mol_name = 'h4_linear'
    num_electrons = 4
    num_orbitals = 8
    dir = f'hamiltonian_gen_test/{mol_name}'
    original_hams = find_files(dir+'/ham_qubit',".data")
    for i, filename in enumerate(original_hams):
        qubit_operator = load_operator(data_directory=dir+'/ham_qubit', file_name=filename, plain_text=True)
        initial_state = SparseVector()
        initial_state.add_entry('00001111', 1)
        arnoldi_evals = arnoldi(qubit_operator, initial_state, 18)
        print("Ensuring that the eigenvalues are close to real: ", np.isclose(arnoldi_evals.imag, 0).all())
        # Cast the eigenvalues to real
        real_evals = arnoldi_evals.real
        lowest_eigenvalue = np.min(real_evals)
        print("Arnoldi ground state energy: ", lowest_eigenvalue)
        ground_state_energy, _ = get_ground_state(get_sparse_operator(qubit_operator))
        print("Ground state energy: ", ground_state_energy)
        print("Difference: ", lowest_eigenvalue - ground_state_energy)
        assert np.isclose(lowest_eigenvalue, ground_state_energy)

def test_arnoldi_hubbard_ham():
    x_dim = 8
    y_dim = 1
    n_sites = x_dim*y_dim
    n_alpha = n_sites//2 + n_sites%2
    n_beta = n_sites - n_alpha

    PHS = True
    PBC = True
    J = 4
    U = 1
    H_fer = fermi_hubbard(x_dim, y_dim, J, U, particle_hole_symmetry=PHS, periodic=PBC)
    qubit_operator = jordan_wigner(H_fer)
    initial_state = SparseVector()
    initial_state.add_entry('10'*n_alpha+'01'*n_beta, 1)
    arnoldi_evals = arnoldi(qubit_operator, initial_state, 16)
    print("Ensuring that the eigenvalues are close to real: ", np.isclose(arnoldi_evals.imag, 0).all())
    # Cast the eigenvalues to real
    real_evals = arnoldi_evals.real
    lowest_eigenvalue = np.min(real_evals)
    print("Arnoldi ground state energy: ", lowest_eigenvalue)
    ground_state_energy, _ = get_ground_state(get_sparse_operator(qubit_operator))
    print("Ground state energy: ", ground_state_energy)
    print("Difference: ", lowest_eigenvalue - ground_state_energy)
    # assert np.isclose(lowest_eigenvalue, ground_state_energy)

if __name__ == '__main__':
    A = QubitOperator('X0 Y1 Z2', 1.0) + QubitOperator('Y3 Z4', 2.0) + QubitOperator(' ', 20.0)
    phi = SparseVector()
    phi.add_entry('00000', 1)
    A_symp = pauli_operator_to_symplectic(A)
    phi_new = apply_Pauli_to_SparseVector(A, phi)
    phi_new_symp = A_symp._apply_1_term_to_SparseVector(phi)
    print(phi_new.data)