import matplotlib.pyplot as plt
import numpy as np
from openfermion.linalg import get_sparse_operator
from openfermion import load_operator
from scipy.sparse import diags
from scipy.sparse import csr_matrix, csc_matrix
from .read_ham import find_files, quick_load

# Take a general sparse matrix as input, and return a new sparse matrix with diagonal elements set to 0
def set_diag_to_zero(sparse_matrix):
    """
    Takes a sparse matrix and returns a new sparse matrix with diagonal elements set to 0.

    Args:
    sparse_matrix (csr_matrix): A scipy CSR sparse matrix.

    Returns:
    csr_matrix: A new sparse matrix with diagonal elements set to 0.
    """
    if not isinstance(sparse_matrix, (csr_matrix, csc_matrix)):
        raise ValueError("Input must be a CSR/CSC sparse matrix")

    # Extract the diagonal elements
    diagonal = sparse_matrix.diagonal()

    # Create a sparse matrix of the diagonal
    diagonal_matrix = diags(diagonal, 0, shape=sparse_matrix.shape)

    # Subtract the diagonal matrix from the original matrix
    new_matrix = sparse_matrix - diagonal_matrix
    return new_matrix


def visualize_complex_matrix(matrix, save_path):
    """
    Visualizes a given matrix using a colormap.

    Args:
    matrix (np.array): A 2D numpy array representing the matrix to be visualized.

    Returns:
    None
    """
    magnitude_matrix = np.abs(matrix)
    plt.imshow(magnitude_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    # plt.title("Matrix Visualization")
    plt.savefig(save_path, format='pdf')
    # plt.show()
    plt.close()

if __name__ == "__main__":

    dir = 'hamiltonian_gen_test/h2o/4e4o'
    load_ham_type = 'fer'
    ham_filenames = find_files(dir+'/ham_'+load_ham_type,".data") # list of filenames ending with ".data"

    for i, filename in enumerate(ham_filenames):
        H_loaded_q = load_operator(data_directory=dir+'/ham_'+load_ham_type, file_name=filename, plain_text=True)
        H_diag_0 = set_diag_to_zero(get_sparse_operator(H_loaded_q))
        visualize_complex_matrix(H_diag_0.toarray(), save_path=dir+'/ham_'+load_ham_type+'/'+filename[:-5]+'.pdf')