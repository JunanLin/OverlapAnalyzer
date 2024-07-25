import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

def get_exp_val_symmetries(v, n_spin_orb):
    """
    Return expectation values of Sz, N, and S^2 operators for the given eigenvectors v (v assumed to be obtained from eigsh).
    """
    from openfermion import sz_operator, number_operator, s_squared_operator, get_sparse_operator
    exp_sz = [v[:,i].T @ get_sparse_operator(sz_operator(int(n_spin_orb//2))) @ v[:,i] for i in range(len(v.T))]
    exp_n = [v[:,i].T @ get_sparse_operator(number_operator(int(n_spin_orb))) @ v[:,i] for i in range(len(v.T))]
    exp_s2 = [v[:,i].T @ get_sparse_operator(s_squared_operator(int(n_spin_orb//2))) @ v[:,i] for i in range(len(v.T))]
    return exp_sz, exp_n, exp_s2

def hf_occupation_list(n_electrons, n_unpaired):
    """
    Generate a list of occupation numbers for the Hartree-Fock state.

    :param n_electrons: the total number of electrons
    :param n_unpaired: the number of unpaired electrons, support aliasing: n_unpaired = 's0' or 't1'
    :return: the occupation list
    """
    if n_unpaired == 's0':
        n_unpaired = 0
    elif n_unpaired == 't1':
        n_unpaired = 2
    # Ensure that both n_electrons and n_unpaired are even numbers
    if n_electrons % 2 != 0 or n_unpaired % 2 != 0:
        raise ValueError("Both n_electrons and n_unpaired must be even numbers.")
    n_paired = n_electrons - n_unpaired
    return list(range(0, n_paired)) + list(range(n_paired, n_paired + 2*(n_unpaired - 1)+1, 2))


def sort_eigen(eigenvalues, eigenvectors):
    # Get the indices that would sort the energy array
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return sorted_eigenvalues, sorted_eigenvectors

def calculate_lowest_eigen(matrix, n, is_hermitian=False):
    if is_hermitian:
        # If matrix is a csc or csr matrix, use eigsh
        if isinstance(matrix, (csc_matrix, csr_matrix)):
            print("Matrix is sparse Hermitian, using eigsh...")
            eigenvalues, eigenvectors = eigsh(matrix, k=n, which='SA')
        else:
            print("Matrix is Hermitian, using eigh...")
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    else:
        # Otherwise, use eig
        print("Matrix is not Hermitian, using eig...")
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_eigenvalues, sorted_eigenvectors = sort_eigen(eigenvalues, eigenvectors)

    # Take the n lowest eigenvalues and their corresponding eigenvectors
    lowest_eigenvalues = sorted_eigenvalues[:n]
    lowest_eigenvectors = sorted_eigenvectors[:, :n]

    return lowest_eigenvalues, lowest_eigenvectors

def count_degeneracies(eigenvals, tolerance=1e-8):
    """
    Count the degeneracies of the eigenvalues.
    Only works for eigenvals sorted.
    """
    # Raise error if the eigenvalues are not sorted from lowest to highest
    if not np.all(np.diff(eigenvals) >= 0):
        raise ValueError("Eigenvalues must be sorted from lowest to highest.")
    # Initialize an empty list to store the counts
    degeneracies = []

    # Iterate over the array
    i = 0
    while i < len(eigenvals):
        count = 1
        # Count the number of similar elements (within the tolerance)
        while i + 1 < len(eigenvals) and np.abs(eigenvals[i] - eigenvals[i + 1]) < tolerance:
            i += 1
            count += 1
        # Append the count to the degeneracies list
        degeneracies.append(count)
        i += 1

    return np.array(degeneracies)

def evals_no_degen(eigenvals, degen):
    """
    Return a list of eigenvalues without degeneracies.
    """
    start = 0
    evals = []
    for i, d in enumerate(degen):
        evals.append(eigenvals[start])
        start += d
    return evals

# def find_subspace_index(degen, index):
#     """
#     Find the subspace index that the given index belongs to, based on the list of degeneracies.

#     :param degen: numpy array of degeneracies
#     :param index: the index to find the subspace for
#     :return: the index labelling the subspace that the given index belongs to
#     """
#     sum_degen = 0
#     for subspace_index, d in enumerate(degen):
#         sum_degen += d
#         if index <= sum_degen-1:
#             return subspace_index
#     return None  # If index is out of the range of the degeneracies

# def vectors_in_subspace(v, degen, original_index):
#     """
#     Return all vectors in v that belong to the same subspace as the vector labelled by original_index.

#     :param v: numpy array of vectors (assumed from eigh/eigsh)
#     :param degen: numpy array of degeneracies
#     :param index: the index to find the subspace for
#     :return: numpy array of vectors in the identified subspace
#     """
#     subspace_idx = find_subspace_index(degen, original_index)
#     if subspace_idx is None:
#         return np.array([])  # Return empty array if index is out of range

#     start = sum(degen[:subspace_idx])  # Start index of the subspace in v
#     end = start + degen[subspace_idx]  # End index (exclusive) of the subspace in v
#     return v[start:end]

def find_subspace_indices(degen, idx):
    start = 0
    for i in range(idx):
        start += degen[i]
    return (start, start+degen[idx])

def vecs_in_subspace(vectors, degen, idx):
    """
    Returns a part of the vectors matrix that corresponds to the subspace with the given index.
    Usage: get_subspace(vectors, degen, 0) returns the vectors corresponding to the first subspace defined by degen, etc.
    """
    start, end = find_subspace_indices(degen, idx)
    return vectors[:, start:end]

def overlap_with_ON_vecs(v, vectors):
    """
    Compute total overlap of vector v with a set of orthonormal vectors.
    :param v: the vector to compute overlap with
    :param vectors: the vectors to compute overlap with, given as a numpy array; each column represents a vector
    """
    overlaps = 0
    for i in range(vectors.shape[1]):
        overlaps += np.abs(np.vdot(v, vectors[:, i])) ** 2
    return overlaps

def overlap_with_vectors(v, vectors, degen):
    """
    Compute a list of overlaps of vector v with a set of vectors, where degen specifies how the vectors are grouped.
    """
    overlaps = []
    start = 0
    for d in degen:
        overlaps.append(overlap_with_ON_vecs(v, vectors[:, start:start+d]))
        start += d
    return overlaps

def orthonormalize(v, degen):
    """
    Orthonormalize the eigenvectors within each degenerate subspace.
    """
    start = 0
    v_ON = np.array([])
    for d in degen:
        # Orthonormalize the subspace using QR decomposition
        Q, _ = np.linalg.qr(v[:, start:start+d])
        v_ON = np.concatenate((v_ON, Q), axis=1) if v_ON.size else Q
        start += d
    return v_ON

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

def Davidson_original(H, ini_guess, tol, n_max = 100):
    """Performs original Davidson diagonalization algorithm in S. Cotton's work. UNFINISHED."""
    b = [ini_guess]
    sigma = []
    for n in range(1,n_max):
        sigma.append(np.dot(H,b[n-1]))

if __name__ == '__main__':
    import numpy as np
    # print(vecs_in_subspace(np.array([[2,1,3],[4,0,1],[3,5,2]]), [2,1], 1))
    # print(evals_no_degen([5,5,3,1,1], [2,1,2]))
    print(find_subspace_indices([2,1,2], 2))

#Hello World