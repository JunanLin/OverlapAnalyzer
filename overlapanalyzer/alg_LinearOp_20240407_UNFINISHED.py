# Import necessary packages
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.sparse.linalg import gmres, spsolve
import time
from openfermion import QubitOperator, get_sparse_operator, load_operator
# For testing purposes
import os
from scipy.sparse.linalg import eigsh
from openfermion import jw_hartree_fock_state
from .iQCC import apply_iQCC_gens_to_state
from .read_ham import find_files, quick_load, load_mol_info

banner = "="*70
class linear_counter():
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

class linear_solver():
    def __init__(self, A_sparse, b, x_guess, P = None):
        self.A_sparse = A_sparse
        self.b = b
        self.x_guess = x_guess
        self.P = P
        self.matvec_count = 0
        self.final_x_each_iter = []

    
    def gmres_solve(self, tol=1e-4, restart=4, maxiter=20):
        def record_current_x(x):
            self.final_x_each_iter.append(x)
    
        def matvec_counted(A, x):
            self.matvec_count += 1
            return A @ x
        # counter_A = linear_counter()
        M = None
        if self.P is not None:
            P_counted = lambda x: matvec_counted(self.P, x)
            # identity_coeff = get_id_coeff_QubitOperator(self.A_precond) # A_precond must be a QubitOperator for now; generalize later!
            # y0 = 1/(identity_coeff) * self.x_guess
            M_x = lambda x: spsolve(P_counted, x) # Solves for A_precond @ y = x; returns y. Initial guess for y is 1/A_id * x where A_id is the identity part of A_precond
            M = LinearOperator(self.A_sparse.shape, M_x)
        # self.matvec_count = 0
        A_operator = LinearOperator(self.A_sparse.shape, matvec=lambda x: matvec_counted(self.A_sparse, x))
        x, info = gmres(A_operator, self.b, x0 = self.x_guess, M=M, atol=tol, restart=restart, maxiter=maxiter, callback=record_current_x, callback_type='x')
        # x, info = gmres(A_operator, self.b, x0 = self.x_guess, M=M, atol=tol, restart=restart, maxiter=maxiter)
        return x, self.matvec_count, self.final_x_each_iter


# def get_diagonal(A):
#     n = A.shape[0]
#     diag = np.zeros(n)
#     for i in range(n):
#         e = np.zeros(n)
#         e[i] = 1
#         diag[i] = A.matvec(e)[i]
#     return diag

def get_diag_part_QubitOperator(H: QubitOperator):
    original_H = H
    # Diagonal part of a QubitOperator is the collection of terms which only contains PauliZ operators, plus the identity
    diag_operator = QubitOperator()
    for term, coeff in original_H.terms.items():
        if all(map(lambda x: x[1] == 'Z', term)):
            if np.isclose(coeff.imag, 0):
                coeff = coeff.real
            diag_operator += QubitOperator(term, coeff)
    return diag_operator

def get_id_coeff_QubitOperator(H):
    """
    Returns the coefficient of the identity part of a QubitOperator
    """
    return H.terms.get((), 0)

def compute_inv_applied_to_state(E_Ritz, H_diag, phi, n_qubits, tol = 1e-2):
    """
    Apply the inverse of H_diag onto phi by solving a linear system: (E_Ritz - H_diag) x = phi
    H_diag is a diagonal QubitOperator
    phi is a numpy array
    Returns a numpy array
    """
    E_Ritz_QubitOperator = QubitOperator('', E_Ritz)
    # Find identity part ([]) of H_diag, extract the coefficient
    identity_coeff = get_id_coeff_QubitOperator(H_diag)

    precond = get_sparse_operator(E_Ritz_QubitOperator - H_diag, n_qubits=n_qubits)
    # Solve the linear system
    solver = linear_solver(precond, phi, 1/(E_Ritz - identity_coeff) * phi) # Here a preconditioner which only involves the identity part is used
    x, niter = solver.gmres_solve(tol = tol, restart=10)
    return x, niter



def test_get_diag_part_QubitOperator():
    H = QubitOperator('X0 Y1 Z2',1.5) + QubitOperator('Z0 Z1', 2.5) + QubitOperator('X0', 3.5) + QubitOperator('', 4.5)
    print(get_diag_part_QubitOperator(H))

def test_get_diag_part_from_file(filename, filepath):
    H = load_operator(filename, filepath, plain_text=True)
    H_diag = get_diag_part_QubitOperator(H)
    print(H_diag)
    print(H_diag.terms.get((), 0))

def test_compute_inv_applied_to_state():
    H = QubitOperator('X0 Y1 Z2',1.5) + QubitOperator('Z0 Z1', 2.5) + QubitOperator('X0', 3.5) + QubitOperator('', 4.5)
    H_diag = get_diag_part_QubitOperator(H)
    E_Ritz = 5
    phi = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    x, niter = compute_inv_applied_to_state(E_Ritz, H_diag, phi, 3)
    x_exact = np.linalg.inv(E_Ritz * np.eye(8) - get_sparse_operator(H_diag, n_qubits=3)) @ phi
    print("Norm of difference: ", np.linalg.norm(x - x_exact))
    print("Number of iterations: ", niter)



def lanczos_lowest_eigenvalue(A, v, tol=1e-8, max_iter=100):
    if not isinstance(A, LinearOperator):
        raise ValueError("A must be a scipy.sparse.linalg.LinearOperator")
    matvec_count = 0
    vecvec_count = 0
    v1 = v / np.linalg.norm(v)
    V = v1.copy()
    vecvec_count += 1
    w1_prime = A @ v1
    matvec_count += 1
    time_now = time.time()
    alpha_1 = np.real(np.vdot(w1_prime, v1))
    # Raise warning if imaginary part of alpha_1 is not negligible
    if abs(np.imag(alpha_1)) > 1e-14:
        print(f"Imaginary part of alpha_1 is {np.imag(alpha_1)}, casting to real; results may not be accurate.")
    vecvec_count += 1
    w1 = w1_prime - alpha_1 * v1
    vecvec_count += 1

    vk_old = v1
    wk_old = w1

    eigenvalues = []
    eigenvectors = []
    iter_number = []
    matvec_list = []
    vecvec_list = []
    method_list = []
    runtime_list = []

    alphas = [alpha_1]
    betas = []
    time_init = time.time()
    for i in range(max_iter):
        beta = np.linalg.norm(wk_old)
        if beta > 0:
            betas.append(beta)
            vk = wk_old / beta
            V = np.column_stack((V, vk))

        wk_prime = A @ vk
        matvec_count += 1
        alpha = np.real(np.vdot(wk_prime, vk))
        if abs(np.imag(alpha)) > 1e-14:
            print(f"Imaginary part of alpha_{i+1} is {np.imag(alpha)}, casting to real; results may not be accurate.")
        vecvec_count += 1

        alphas.append(alpha)
        wk = wk_prime - alpha * vk - beta * vk_old
        vecvec_count += 2

        eig_val, eig_vec = eigh_tridiagonal(alphas, betas, select='i', select_range=(0,0))
        time_now = time.time()
        eigenvalues.append(eig_val[0])
        eigenvectors.append(V @ eig_vec[:, 0])
        iter_number.append(i+1)
        matvec_list.append(matvec_count)
        vecvec_list.append(vecvec_count)
        method_list.append("Lanczos")
        runtime_list.append(time_now - time_init)

        if len(eigenvalues) >= 2 and abs(eigenvalues[i] - eigenvalues[i-1]) < tol:
            break

        vk_old = vk
        wk_old = wk

    return {"method": method_list, "eigenvalues": eigenvalues, "eigenvectors": eigenvectors, "iterations": iter_number, "matvec_list": matvec_list, "vecvec_list": vecvec_list, "runtime_list": runtime_list}

def davidson_lowest_eigenvalue(A, v, diag, n_qubits, tol=1e-4, max_iter=20):
    if not isinstance(A, LinearOperator):
        raise ValueError("A must be a scipy.sparse.linalg.LinearOperator")
    matvec_count = 0
    vecvec_count = 0
    if len(v.shape)>1:
        v=v.reshape(-1)
    # First iteration
    v1 = v / np.linalg.norm(v)
    vecvec_count += 1
    V = np.array([v1.copy()])
    w = A @ v1
    matvec_count += 1
    W = np.array([w.copy()])
    # Create a variable to store the subspace Hamiltonian H, start from a 1-by-1 matrix then expand it iteratively
    H = np.zeros((1, 1))
    E_save = np.real(np.vdot(v, w))
    H[0, 0] = E_save
    vecvec_count += 1
    
    eigenvalues = []
    eigenvectors = []
    iter_number = []
    matvec_list = []
    vecvec_list = []
    method_list = []
    runtime_list = []

    time_init = time.time()
    for i in range(max_iter):
        # Expand the subspace Hamiltonian H by appending new rows and columns
        if i > 0:
            H = np.pad(H, ((0, 1), (0, 1)), 'constant')
            for j in range(i):
                H[i, j] = np.real(np.vdot(V[j], W[i]))
                H[j, i] = np.real(np.vdot(V[i], W[j]))
                vecvec_count += 2 
            H[i, i] = np.real(np.vdot(V[i], W[i]))
            vecvec_count += 1
        E_Ritz, c = eigh(H)
        # Ensure eigenvalues are sorted
        # sorted_index = list(reversed(E_Ritz.argsort()[::-1]))
        # E_Ritz = E_Ritz[sorted_index]
        # c = c[:, sorted_index]

        eigenvalues.append(E_Ritz[0])
        eigenvectors.append(V.T @ c[:, 0])
        iter_number.append(i)
        matvec_list.append(matvec_count)
        vecvec_list.append(vecvec_count)
        method_list.append("Davidson")
        time_now = time.time()
        runtime_list.append(time_now - time_init)

        if i > 0 and abs(E_Ritz[0] - E_save) < tol:
            break

        E_save = E_Ritz[0]

        # W has shape (i+1, N); c[:,0] has shape (i+1). Need to transpose W before multiplying
        r = W.T @ c[:, 0]- (E_Ritz[0] * V).T @ c[:, 0]
        vecvec_count += (2*i+3)
        # Apply the preconditioner
        # precond = 1 / (E_Ritz[0] - diag)
        # r = (precond * r)
        r, n_iter = compute_inv_applied_to_state(E_Ritz[0], diag, r, n_qubits)
        matvec_count += n_iter
        # vecvec_count += 1 # vecvec cannot be counted in current implementation
        # Orthogonalize r with respect to all existing column vectors in V
        r = r - V.T @ (V @ r)
        vecvec_count += (2*i+3)
        # Normalize r
        r = r / np.linalg.norm(r) # r has shape (N,) up to this step
        vecvec_count += 1
        V = np.concatenate((V, r[:, np.newaxis].T))# To get the dimension right, r needs to have shape (1, N)
        
        w = A @ r # w has shape (N,)
        matvec_count += 1
        W = np.concatenate((W, w[:, np.newaxis].T)) 
    return {"method": method_list, "eigenvalues": eigenvalues, "eigenvectors": eigenvectors, "iterations": iter_number, "matvec_list": matvec_list, "vecvec_list": vecvec_list, "runtime_list": runtime_list}

def shifted_lanczos(A, v, zi, tol=1e-8, max_iter=100):
    if not isinstance(A, LinearOperator):
        raise ValueError("A must be a scipy.sparse.linalg.LinearOperator")
    
    matvec_count = 0
    vecvec_count = 0
    expectation_values = []
    iter_number = []
    matvec_list = []
    vecvec_list = []
    method_list = []
    runtime_list = []

    # Normalization of v and initial setups
    v1 = v / np.linalg.norm(v)
    vecvec_count += 1
    s1 = A @ v1
    matvec_count += 1
    alpha_1 = np.vdot(s1, v1)
    vecvec_count += 1

    # Initialize variables for all shifts
    c = {shift: np.vdot(v, v) for shift in zi}
    vecvec_count += len(zi)
    pi = {shift: 1 / (shift - alpha_1) for shift in zi}
    L = {shift: c[shift] / (shift - alpha_1) for shift in zi}
    expectation_values.append([L[shift] for shift in zi])
    iter_number.append(1)
    matvec_list.append(matvec_count)
    vecvec_list.append(vecvec_count)
    method_list.append("Shifted Lanczos")

    L_old = L.copy()

    vk = v1
    sk = s1
    alpha_k = alpha_1

    time_init = time.time()
    for k in range(2, max_iter + 2):
        tk = sk - alpha_k * vk
        vecvec_count += 1
        beta_k = np.linalg.norm(tk)
        vecvec_count += 1

        # if beta_k < tol:  # Convergence check based on deltak
        #     break

        vk_next = tk / beta_k
        sk_next = A.matvec(vk_next) - beta_k * vk
        matvec_count += 1
        vecvec_count += 1
        alpha_k_next = np.real_if_close(np.vdot(sk_next, vk_next))
        vecvec_count += 1

        # Update L, fi, and c for all shifts and check for convergence
        converged = False
        for shift in zi:
            tki = (beta_k ** 2 * pi[shift])
            pi_next = 1 / (shift - alpha_k_next - tki)
            ci_next = c[shift] * tki * pi[shift]
            L_old[shift] = L[shift]
            L[shift] += ci_next * pi_next
            vecvec_count += 1

            # Check if the change in L is below the tolerance for all shifts
            # If any shift is above tol, then we'll go for another iteration
            if abs(L[shift] - L_old[shift]) < tol:
                converged = True

            # Update values for next iteration
            pi[shift] = pi_next
            c[shift] = ci_next
        
        expectation_values.append([L[shift] for shift in zi])
        iter_number.append(k)
        matvec_list.append(matvec_count)
        vecvec_list.append(vecvec_count)
        method_list.append("Parallel Lanczos")
        time_now = time.time()
        runtime_list.append(time_now - time_init)
        if converged:  # If converged for all shifts, break
            break

        # Update vk and sk for the next iteration
        vk = vk_next
        sk = sk_next
        alpha_k = alpha_k_next

    return {"method": method_list, "expectation_values": expectation_values, "iterations": iter_number, "matvec_list": matvec_list, "vecvec_list": vecvec_list, "runtime_list": runtime_list}

# Write a function to implement the conjugate gradient method to solve a linear system Ax=b, keeping track of the number of matrix-vector and vector-vector multiplications after each iteration as above.
def conjugate_gradient(A, b, x0=None, tol=1e-8, max_iter=100):
    # Generated by CoPilot, not tested yet
    if not isinstance(A, LinearOperator):
        raise ValueError("A must be a scipy.sparse.linalg.LinearOperator")

    matvec_count = 0
    vecvec_count = 0
    if x0 is None:
        x0 = np.zeros(A.shape[0])
    r0 = b - A @ x0
    matvec_count += 1
    p0 = r0
    vecvec_count += 1
    xk = x0
    rk = r0
    pk = p0

    # Initialize lists to store the number of matrix-vector and vector-vector multiplications after each iteration
    matvec_list = [matvec_count]
    vecvec_list = [vecvec_count]
    method_list = ["Conjugate Gradient"]

    for k in range(1, max_iter + 1):
        alpha_k = np.vdot(rk, rk) / np.vdot(pk, A @ pk)
        vecvec_count += 2
        xk_next = xk + alpha_k * pk
        vecvec_count += 1
        rk_next = rk - alpha_k * A @ pk
        matvec_count += 1
        beta_k = np.vdot(rk_next, rk_next) / np.vdot(rk, rk)
        vecvec_count += 2
        pk_next = rk_next + beta_k * pk
        vecvec_count += 1

        # Update values for the next iteration
        xk = xk_next
        rk = rk_next
        pk = pk_next

        # Append the number of matrix-vector and vector-vector multiplications to the lists
        matvec_list.append(matvec_count)
        vecvec_list.append(vecvec_count)
        method_list.append("Conjugate Gradient")
        # Check for convergence
        if np.linalg.norm(rk) < tol:
            break

    return {"method": method_list, "iterations": k, "matvec_list": matvec_list, "vecvec_list": vecvec_list}

def gmres_1_shift(H_sparse, v, z, x0=None, M=None, tol=1e-8, max_iter=100):
    def matvec_shifted(x):
        matvec_count += 1
        return z*x - H_sparse @ x
    def record_iteration_result(x):
        expectation_values.append(np.vdot(v, x))
        # iter_number.append(k)
        matvec_list.append(matvec_count)
        vecvec_list.append(vecvec_count)
        method_list.append("GMRES")
    matvec_count = 0
    vecvec_count = 0
    expectation_values = []
    iter_number = []
    matvec_list = []
    vecvec_list = []
    method_list = []

    A_operator = LinearOperator(H_sparse.shape, matvec=matvec_shifted)

    return {"method": method_list, "quadratic_forms": {}, "matvec_list": matvec_list}

def fix_point(zi, H, b, x0=None, tol=1e-6, max_iter=100):
    # Solves (zI-H)x=b using the fixed-point iteration method
    # if not isinstance(H, LinearOperator):
    #     raise ValueError("A must be a scipy.sparse.linalg.LinearOperator")
    if x0 is None:
        x0 = [np.zeros(H.shape[0]) for z in zi]
    elif len(x0) != len(zi):
        raise ValueError("x0 must have the same length as zi")
    
    if x0[0].ndim == 1:
        print("x0 must be a list of 2D arrays, attempting conversion")
        x0_2D = [[] for _ in range(len(x0))]
        for i in range(len(x0)):
            x0_2D[i] = x0[i].reshape(-1, 1) # Convert 1D arrays to 2D column vectors
        x0 = x0_2D
    
    xk = x0
    iter_number = [0]
    matvec_list = [0]
    vecvec_list = [0]
    method_list = ["Fixed Point"]
    expectation_values = [[np.vdot(b, x0[i]) for i, _ in enumerate(zi)]]
    matvec_count = 0
    vecvec_count = 0
    xk_next = [np.zeros(H.shape[0]) for z in zi]
    for k in range(max_iter):
        exp_values_per_iter = []
        for i, z in enumerate(zi):
            xk_next[i] = 1/z * (H @ xk[i] + b)
            matvec_count += 1
            vecvec_count += 1
            exp_values_per_iter.append(np.vdot(b, xk_next[i]))
            xk[i] = xk_next[i]
        expectation_values.append(exp_values_per_iter)
        matvec_list.append(matvec_count)
        vecvec_list.append(vecvec_count)
        method_list.append("Fixed Point")
        iter_number.append(k+1)
    return {"method": method_list, "expectation_values": expectation_values, "iterations": iter_number, "matvec_list": matvec_list, "vecvec_list": vecvec_list}


if __name__ == '__main__':
    from numerical_contour_integration import get_contour
    from numerical_contour_integration import sum_gauss_points
    from openfermion import jw_configuration_state
    import pickle

    
    state_to_eval = 'iQCC' # hf or iQCC

    # dir = 'hamiltonian_gen_test/'+molecule
    current_dir = os.path.dirname(__file__)
    ham_directory = os.path.join(current_dir, 'gen_hams')
    num_electron, num_spin_orb = (8, 16)
    n_qubits = num_spin_orb

    spin = 't1'
    if spin == 't1':
        original_hams = find_files(os.path.join(ham_directory,'ham_qubit'),"t1.data")
        phi_ini = jw_configuration_state([i for i in range(num_electron-1)] + [num_electron], n_qubits) # 2-alpha state
        # Beta state has higher energy by construction; can be constructed similarly

    elif spin == 's0':
        original_hams = find_files(os.path.join(ham_directory,'ham_qubit'),"s0.data")
        phi_ini = jw_configuration_state([i for i in range(num_electron)], n_qubits) # HF state


    for i, filename in enumerate(original_hams):
        print("Running cost comparison for: ", filename)
        # phi = quick_load(dir+'/ham_dressed',filename[:-5] + ".pkl")['out_state'] # a numpy ndarray
        H_QubitOperator = load_operator(data_directory=os.path.join(ham_directory,'ham_qubit'), file_name=filename, plain_text=True)
        H_diag_QubitOperator = get_diag_part_QubitOperator(H_QubitOperator)
        # H_sparse = get_sparse_operator(H_QubitOperator)
        with open('H_sparse.pkl', 'rb') as f:
            H_sparse = pickle.load(f)


        # H_linear = LinearOperator(H_sparse.shape, matvec=lambda x: H_sparse @ x)
        # Load exact ground state and energy

        # E0, psi_0 = quick_load(dir+'/ham_fer',filename[:-5] + "_Exact_GS_vec.pkl")
        # w, v = eigsh(H_sparse, k=2, which='SA', v0 = phi_ini)
        with open('temp_eigenvalues.pkl', 'rb') as f:
            w = pickle.load(f)
            v = pickle.load(f)
        E0 = w[0]
        psi_0 = v[:, 0]
        if state_to_eval == 'hf':
            phi = phi_ini
        elif state_to_eval == 'iQCC':
            # Load iQCC results
            iQCC_results = quick_load(os.path.join(ham_directory,'iQCC_'+spin),filename[:-5] + "_iQCC.pkl")
            # iQCC_results['num_qubits'] = num_spin_orb
            iQCC_energy = iQCC_results['energies'][-1][0]
            # Get iQCC state
            phi = apply_iQCC_gens_to_state(iQCC_results, phi_ini)
        overlap_exact = np.abs(np.vdot(psi_0, phi)) ** 2
        print("Exact phi-ground state overlap: ", overlap_exact)
        
        contour = get_contour(w[0],(w[1]-w[0])/2, 8)

        # Calculate lowest eigenvalue using GMRES
        psi_0 = psi_0.reshape(-1)
        phi = phi.reshape(-1)
        if state_to_eval == 'iQCC':
            E_state = iQCC_energy
        else:
            E_state = np.vdot(phi, H_sparse @ phi)

        matvec_count = 0
        matvec_list = []
        output_each_iter = []
        computed_QF = []
        # First point
        z = contour["Points"][0]


        # A_sparse = get_sparse_operator(QubitOperator('', z) - H_QubitOperator)
        A_sparse = z*np.ones(H_sparse.shape[0]) - H_sparse
        # A_sparse = diags(z*np.ones(H_sparse.shape[0])) - H_sparse
        # M_diag = z * np.ones(H_sparse.shape[0]) - H_sparse.diagonal()
        # M_sparse = diags(M_diag)
        # M_inv_diag = 1/M_diag
        # M_inv_sparse = diags(M_inv_diag)
        solver = linear_solver(A_sparse, phi, 1/(z - get_id_coeff_QubitOperator(H_QubitOperator)) * phi, P=QubitOperator('', z) - H_diag_QubitOperator)
        print("Finished Constructing Linear Solver.")
        y_est_1, n_matvec, y_vecs_1 = solver.gmres_solve(tol = 1e-2, restart=10)
        output_each_iter.append(y_vecs_1)
        matvec_count += n_matvec
        computed_QF.append(np.vdot(phi, y_est_1))
        for i in range(1, len(contour["Points"])):
            z = contour["Points"][i]
            # A_sparse = get_sparse_operator(QubitOperator('', z) - H_QubitOperator)
            A_sparse = z*np.ones(H_sparse.shape[0]) - H_sparse
            # A_sparse = diags(z*np.ones(H_sparse.shape[0])) - H_sparse
            # M_diag = z * np.ones(H_sparse.shape[0]) - H_sparse.diagonal()
            # M_sparse = diags(M_diag)
            # M_inv_diag = 1/M_diag
            # M_inv_sparse = diags(M_inv_diag)
            solver = linear_solver(A_sparse, phi, (contour["Points"][i-1] - E_state)/(z-E_state) * y_est_1, P=QubitOperator('', z) - H_diag_QubitOperator)
            y_est, n_matvec, y_vecs = solver.gmres_solve(tol = 1e-2, restart=10)
            output_each_iter.append(y_vecs)
            matvec_count += n_matvec
            computed_QF.append(np.vdot(phi, y_est))
        overlaps = sum_gauss_points(contour, computed_QF)
        overlap_error = (overlap_exact - np.array(overlaps))/overlap_exact
        result = ["GMRES", overlap_exact, overlap_error, "NA", matvec_count, "NA"]
        print(result)
        # with open(os.path.join(ham_directory,filename[:-5]+'_costs_'+state_to_eval+'.csv'), 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(result)
        print("GMRES completed.")