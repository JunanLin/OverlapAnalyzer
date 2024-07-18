import numpy as np
import datetime
from numpy import real_if_close
from openfermion import load_operator, get_sparse_operator, jw_hartree_fock_state, Davidson, DavidsonOptions
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from .Hdot_count import CountedLinearOperator
from .read_ham import find_files, extract_numbers_from_filenames, quick_load
from .eigen import calculate_lowest_eigen

def count_degeneracies(arr, tolerance=1e-10):
    # Initialize an empty list to store the counts
    degeneracies = []

    # Iterate over the array
    i = 0
    while i < len(arr):
        count = 1
        # Count the number of similar elements (within the tolerance)
        while i + 1 < len(arr) and np.abs(arr[i] - arr[i + 1]) < tolerance:
            i += 1
            count += 1
        # Append the count to the degeneracies list
        degeneracies.append(count)
        i += 1

    return np.array(degeneracies)


def find_subspace_index(degen, index):
    """
    Find the subspace index that the given index belongs to, based on the list of degeneracies.

    :param degen: numpy array of degeneracies
    :param index: the index to find the subspace for
    :return: the subspace index (starting from 0)
    """
    sum_degen = 0
    for i, d in enumerate(degen):
        sum_degen += d
        if index <= sum_degen:
            return i
    return None  # If index is out of the range of the degeneracies

def vectors_in_subspace(v, degen, index):
    """
    Return all vectors in v that belong to the subspace where the given index lies.

    :param v: numpy array of vectors
    :param degen: numpy array of degeneracies
    :param index: the index to find the subspace for
    :return: numpy array of vectors in the identified subspace
    """
    subspace_idx = find_subspace_index(degen, index)
    if subspace_idx is None:
        return np.array([])  # Return empty array if index is out of range

    start = sum(degen[:subspace_idx])  # Start index of the subspace in v
    end = start + degen[subspace_idx]  # End index (exclusive) of the subspace in v
    return v[start:end]

def orthonormal_basis(vectors):
    """Check if the given vectors are orthonormal. If not, return an orthonormal basis."""
    # Check if the vectors are orthonormal
    if np.allclose(np.eye(vectors.shape[0]), np.dot(vectors, vectors.T)):
        return vectors

    # If not, use Gram-Schmidt to find an orthonormal basis
    basis = []
    for v in vectors:
        w = v - projection(v, basis)
        if np.linalg.norm(w) > 1e-10:
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)

def projection(v, basis):
    """Compute the projection of v onto the subspace spanned by the basis vectors."""
    proj = np.zeros_like(v, dtype=np.complex_)
    for w in basis:
        proj += np.vdot(w, v) / np.vdot(w, w) * w
    return proj

def residue_vector(v, basis):
    """Compute the residue vector of v with respect to the subspace spanned by basis vectors."""
    proj = projection(v, basis)
    return v - proj

# Example usage
if __name__ == "__main__":
    molecule = 'h2o/4e4o'
    num_electrons = 4
    num_orbitals = 8 # Automate later

    dir = 'hamiltonian_gen_test/'+molecule
    original_hams = find_files(dir+'/ham_fer',".data")
    hf_state = jw_hartree_fock_state(4, 8)
    for i, filename in enumerate(original_hams):
        geometry = extract_numbers_from_filenames(original_hams)[i]
        phi = quick_load(dir+'/ham_dressed',filename[:-5] + ".pkl")['out_state'] # a numpy ndarray
        print("Shape of phi: ", phi.shape)
        phi_rand = np.random.rand(phi.shape[0])
        phi_rand = phi_rand/np.linalg.norm(phi_rand)
        A_sparse = get_sparse_operator(load_operator(data_directory=dir+'/ham_fer', file_name=filename, plain_text=True))
        # Compute first 10 eigenvalues and eigenvectors of A_sparse
        A_counted = CountedLinearOperator(A_sparse)
        w_ap, v_ap = eigsh(A_counted, k=10, which='SA')
        print("Matvec count: ", A_counted.counter)
        # Compute a sparse matrix that is equal to the diagonal of A_sparse
        A_diag_vec = A_sparse.diagonal()
        print("Type and shape of hf_state: ", type(hf_state), hf_state.shape)
        # Convert A_sparse to dense, and solve for lowest eigenvalue and eigenvector
        A_dense = A_sparse.todense()
        w, v = calculate_lowest_eigen(A_dense, 10, is_hermitian=True)
        v = np.array(v)
        # print("Shape of v: ", v.shape)
        w = real_if_close(w)
        print("Lowest eigenvalues, dense: ", w)

        # eigen_num = 1
        # Davidson_max_subspace = np.array([3, 4, 5, 6, 7, 8])
        # for j, n_max in enumerate(eigen_num * Davidson_max_subspace):
        #     A_counted = CountedLinearOperator(A_sparse)
        #     davidson_options = DavidsonOptions(max_subspace = Davidson_max_subspace[j], eps=10.0**(-Davidson_max_subspace[j]+1))
        #     davidson_solver = Davidson(linear_operator = A_counted, linear_operator_diagonal = A_diag_vec, options = davidson_options)
        #     start_time = datetime.datetime.now()
        #     success, w_ap, v_ap = davidson_solver.get_lowest_n(n_lowest = 1, initial_guess=np.array([hf_state]).T)
        #     end_time = datetime.datetime.now()
        #     runtime = end_time - start_time
        #     print(f"Eigenvalue error: {np.abs(w_ap[eigen_num - 1] - w[eigen_num - 1])}, matvec count: {A_counted.counter}")
        #     print("Runtime: ", runtime.total_seconds())