from scipy.sparse.linalg import LinearOperator

# 20231113: haven't been able to get a countable LinearQubitOperator
# Also, LinearQubitOperator seems not to work with eigsh and other linear solvers: gives incorrect eigenstates

from scipy.sparse.linalg import gmres
from openfermion import (
    load_operator,
    get_sparse_operator,
)
from overlapanalyzer.read_ham import find_files, quick_load


class CountedLinearOperator(LinearOperator):
    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        self.dtype = A.dtype
        self.counter = 0  # Initialize counter to 0
        super(CountedLinearOperator, self).__init__(dtype=self.dtype, shape=self.shape)

    def _matvec(self, x):
        self.counter += 1  # Increment counter on each multiplication
        return self.A @ x

class GmresResolventOp(LinearOperator):
    def __init__(self, A, z):
        self.A = A
        self.z = z
        self.shape = A.shape
        self.dtype = A.dtype
        self.counter = 0  # Initialize counter to 0
        super(GmresResolventOp, self).__init__(dtype=self.dtype, shape=self.shape)

    def _matvec(self, x):
        self.counter += 1  # Increment counter on each multiplication
        return self.z * x - self.A @ x
    
if __name__ == "__main__":
    dir = 'hamiltonian_gen_test/h2o/4e4o'
    original_hams = find_files(dir+'/ham_fer',".data")
    for i, filename in enumerate(original_hams):
        phi = quick_load(dir+'/ham_dressed',filename[:-5] + ".pkl")['out_state'] # a numpy ndarray
        A_sparse = get_sparse_operator(load_operator(data_directory=dir+'/ham_fer', file_name=filename, plain_text=True))
        my_op = GmresResolventOp(A_sparse, 1)
        x_ap, info = gmres(my_op, phi)
        print("Exit code: ", info)
        print("H-dot count: ", my_op.counter)