import matplotlib.pyplot as plt
import numpy as np
from openfermion.linalg import get_sparse_operator
from openfermion import load_operator
from scipy.sparse import diags
from scipy.sparse import csr_matrix, csc_matrix
from overlapanalyzer.read_ham import find_files, quick_load
from overlapanalyzer.polynomial_estimates import chebval, E_to_x, an_list
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


def visualize_chebyshev(Emin, Emax, E0, E1, Ec=None, chebyshev_degree=10):
    x_values = np.linspace(-1, 1, 400)
    plt.figure(figsize=(10, 8))
    if Ec is None:
        Ec = (E0+E1)/2
    x0 = E_to_x(E0, Emin, Emax)
    x1 = E_to_x(E1, Emin, Emax)
    xc = E_to_x(Ec, Emin, Emax)
    # overlap_est_Chebyshev(-1.8977806459898725*2,1.1465506236600063*2,-1.8813605029470586, hlist)
    cheb_coeffs = an_list(xc, chebyshev_degree)
    for n in range(1, chebyshev_degree):  # Starting from 1 because cheb_coeffs[:0] would be an empty slice
        y_values = chebval(x_values, cheb_coeffs[:n])
        plt.plot(x_values, y_values, label=f'n = {n-1}')
    plt.axvline(x=x0, color='r', linestyle=':', label='x0')
    plt.axvline(x=xc, color='g', linestyle=':', label='xc')
    plt.axvline(x=x1, color='b', linestyle=':', label='x1')

    plt.title('Chebyshev Approximation of Step Function')
    plt.xlabel('x')
    # plt.ylabel('Chebval(x, cheb_coeffs[:n])')
    plt.legend()
    plt.grid(True)
    # plt.savefig("Chebyshev_polynomials.pdf")
    plt.show()

if __name__ == "__main__":

    dir = 'hamiltonian_gen_test/h2o/4e4o'
    load_ham_type = 'fer'
    ham_filenames = find_files(dir+'/ham_'+load_ham_type,".data") # list of filenames ending with ".data"

    for i, filename in enumerate(ham_filenames):
        H_loaded_q = load_operator(data_directory=dir+'/ham_'+load_ham_type, file_name=filename, plain_text=True)
        H_diag_0 = set_diag_to_zero(get_sparse_operator(H_loaded_q))
        visualize_complex_matrix(H_diag_0.toarray(), save_path=dir+'/ham_'+load_ham_type+'/'+filename[:-5]+'.pdf')