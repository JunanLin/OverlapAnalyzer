import math
import os
# import tequila as tq
import numpy as np
import pickle
import json
from scipy.linalg import ishermitian
from scipy.sparse import csr_matrix, csc_matrix, issparse
import scipy.sparse as sparse
import scipy.optimize as optimize
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import gmres, bicgstab, spsolve, norm, inv, spilu, LinearOperator
from overlapanalyzer.contour_integration import numerical_contour_integration

def save_dict(saving_dict, fileDir, filename, save_pkl=False):
    '''
    Saves a target dictionary in two formats:
    - .pkl: Pickle format
    - .json: JSON format
    The function removes elements in the dictionary that are not serializable before saving as a .json file.
    '''
    if save_pkl:
        with open(os.path.join(fileDir, filename+ ".pkl"), 'wb') as f:
            pickle.dump(saving_dict, f)
    # Remove the eigen_states from saving_dict, and save another copy of saving_dict using json
    keys_list = list(saving_dict.keys())
    for key in keys_list:
        if not isinstance(saving_dict[key],(dict, list, tuple, str, int, float, bool, bytes, type(None))):
            saving_dict.pop(key)
    with open(os.path.join(fileDir, filename + ".json"), 'w') as f:
        json.dump(saving_dict, f)

def convert_mol_data_to_xyz_format(mol_data):
    '''
    Convert nuclear geometry list to .xyz format.
    '''

    xyz_str = ''
    for atom in mol_data:
        xyz_str += atom[0] +' ' + ' '.join([f"{coord:.10f}" for coord in atom[1]]) +'\n'

    return xyz_str

def get_molecular_data(mol, geometry, xyz_format=False):
    '''
    Generate the molecular data of the specified molecule
    '''
    if mol == 'h2':
        mol_data = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometry]]
        ]
    elif mol == 'lih':
        mol_data = [
            ['Li', [0, 0, 0]],
            ['H', [0, 0, geometry]]
        ]
    elif mol == 'beh2':
        mol_data = [
            ['Be', [0, 0, 0]],
            ['H', [0, 0, geometry]],
            ['H', [0, 0, -geometry]]
        ]
    elif mol == 'h2o':
        # Giving symmetrically stretch H2O. ∠HOH = 107.6°
        # Geometry is distance between H-O
        angle = 107.6 / 2
        angle = math.radians(angle)
        x = geometry * math.sin(angle)
        y = geometry * math.cos(angle)
        mol_data = [
            ['O', [0, 0, 0]],
            ['H', [-x, y, 0]],
            ['H', [x, y, 0]]
        ]
    elif mol == 'n2':
        mol_data = [
            ['N', [0, 0, 0]],
            ['N', [0, 0, geometry]]
        ]
    elif mol == 'h4':
        # Angular stretch between two neighboring H. R = 1.738 Å fixed radius from center to each H.
        # Geometry is the angle of the circular sector H-origin-H.
        R = 1.738
        angle = math.radians(geometry/2)                
        x = R*math.cos(angle)
        y = R*math.sin(angle)
        mol_data = [
            ['H', [x, y, 0]],
            ['H', [x, -y, 0]],
            ['H', [-x, y, 0]],
            ['H', [-x, -y, 0]]
        ]
    elif mol == 'h4_linear':
        # Linear stretch between two neighboring H. R = 1.738 Å fixed radius from center to each H.
        # Geometry is the distance between two H.
        mol_data = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometry]],
            ['H', [0, 0, 2*geometry]],
            ['H', [0, 0, 3*geometry]]
        ]
    elif mol == 'nh3':
        bondAngle = 107
        bondAngle = math.radians(bondAngle)
        cos = math.cos(bondAngle)
        sin = math.sin(bondAngle)

        # The idea is second and third vecctor dot product is cos(angle) * geometry^2.
        thirdyRatio = (cos - cos**2) / sin
        thirdxRatio = (1 - cos**2 - thirdyRatio**2) ** (1/2)
        mol_data = [
            ['H', [0.0, 0.0, geometry]],
            ['H', [0.0, sin * geometry, cos * geometry]],
            ['H', [thirdxRatio * geometry, thirdyRatio * geometry, cos * geometry]],
            ['N', [0.0, 0.0, 0.0]]
            ]       
    elif mol == 'h2x2':            
        r0 = 0.741
        mol_data = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, r0]],
            ['H', [0, 0, r0+geometry]],
            ['H', [0, 0, 2*r0+geometry]] 
        ]
    else:
        raise(ValueError(mol, 'Unknown moleucles given'))

    if xyz_format:
        return convert_mol_data_to_xyz_format(mol_data)
    else:
        return mol_data

# def build_molecule_tq(mol, R, basis = 'sto-3g', active = None):
#     xyz = get_molecular_data(mol, R)
#     molecule = tq.quantumchemistry.Molecule(geometry=xyz, basis_set = basis , active_orbitals = active)
#     return molecule


def non_zero_ratio(sparse_matrix):
    nonzero_elements = sparse_matrix.nnz
    total_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]
    sparsity = (nonzero_elements / total_elements) 
    return sparsity

# ----------------------------------------------------------------------------------------------
# Some utility functions to deal with vectors
def print_nonzero_elements(vector, num_elements=None):
    # Get the indices of non-zero elements
    nonzero_indices = np.nonzero(vector)[0]
    
    # Get the non-zero elements
    nonzero_elements = vector[nonzero_indices]
    
    # Create a list of pairs (element, binary position)
    element_position_pairs = [(element, bin(index)) for index, element in zip(nonzero_indices, nonzero_elements)]
    
    # Sort the pairs by the absolute value of the elements in descending order
    element_position_pairs.sort(key=lambda pair: -abs(pair[0]))
    
    # Print the pairs
    for i, pair in enumerate(element_position_pairs):
        if num_elements is not None and i >= num_elements:
            break
        print(pair)

def process_vector(vector, mag_threshold = 1e-10, angle_threshold=1e-4):
    """
    Takes a vector and returns a new vector obtained by multiplying the original vector by a complex phase.
    The phase is chosen such that the first non-zero element of the new vector is real.
    If the new vector is close enough to a real vector, then the new vector is returned.
    Otherwise, the original vector is returned.
    """
    # Get the non-zero elements whose magnitude is above the threshold
    nonzero_elements = vector[np.abs(vector) > mag_threshold]
    
    if len(nonzero_elements) == 0:
        print("All elements are zero or close to zero, returning original vector.")
        return vector
    
    # Compute the angle of the first non-zero element
    angle = np.angle(nonzero_elements[0])
    
    # Apply the "reverse angle" to all elements in nonzero_elements
    restored_elements = nonzero_elements * np.exp(-1j * angle)
    print("Restored elements: ", restored_elements)
    
    # Check if the resulting vector is close enough to a real vector
    if np.allclose(np.imag(restored_elements), 0, atol=angle_threshold):
        # If yes, return the real vector
        print("The vector is a real vector multiplied by a complex phase, returning real vector.")
        output_vector = np.real(vector * np.exp(-1j * angle))
        # Truncate components of output_vector that are below the threshold
        output_vector[np.abs(output_vector) < mag_threshold] = 0
        # Return normalized vector
        return output_vector / np.linalg.norm(output_vector)
    else:
        # If no, print a message and return the original vector
        print("The vector is not a real vector multiplied by a complex phase, returning original vector.")
        return vector
# def gen_basis_vec(dim, pos):
#     vec = np.zeros(dim)
#     vec[pos] = 1
#     return vec

def exp_val_higher_moment(H, v, k, return_all=True):
    '''
    Calculates expectation value of <v|H^k|v>.
    Returns all moments [H^0, ..., H^k] if return_all is True.
    '''
    def is_hermitian(H, eps=1e-12):
        if issparse(H):
            conj_transpose = H.conj().T
            return (H - conj_transpose).max() < eps
        else:
            return ishermitian(H) # Use scipy's ishermitian function for dense matrices
    
    if is_hermitian(H):
        print("Hermitian matrix H, using efficient method to compute moments.")
        n_evals = k//2 + 1
        vecs = {0: v}
        moments = np.array([np.real_if_close(np.vdot(v, v))])
        for i in range(n_evals):
            vecs[i+1] = H @ vecs[i]
        for i in range(1, k+1):
            if i % 2 == 0:
                moments = np.append(moments,np.real_if_close(np.vdot(vecs[i//2], vecs[i//2])))
            else:
                moments = np.append(moments, np.real_if_close(np.vdot(vecs[i//2], vecs[i//2+1])))
        return moments if return_all else moments[-1]
    else:
        print("Non-Hermitian matrix H, using general method to compute moments.")
        vk = v
        moments = [np.vdot(v, v)]
        for _ in range(k):
            vk = H @ vk
            moments.append(np.vdot(v, vk))
        return moments if return_all else moments[-1]

def exp_val(H, eig):
    '''Calculates expectation value of <eig|H|eig>.'''
    return exp_val_higher_moment(H, 1, eig)

def exp_val_sq(H, eig):
    '''Calculates expectation value of <eig|H^2|eig>.'''
    return exp_val_higher_moment(H, 2, eig)

def overlap_sq(v1, v2):
        return np.abs(np.vdot(v1, v2)) ** 2 # If v1 is complex, then vdot takes care of the complex conjugate

def general_func_exp_val(H, eig, func):
    if issparse(H):
        H_dense = H.todense()
        w, v = np.linalg.eigh(H_dense)
        func_H_dense = np.dot(v, np.dot(np.diag(func(w)), np.conjugate(v.T)))
        #return np.dot(np.conjugate(eig.T), np.dot(func_H_dense, eig))
        return np.conjugate(eig.T) @ func_H_dense @ eig
    else:
        w, v = np.linalg.eigh(H)
        func_H = np.dot(v, np.dot(np.diag(func(w)), np.conjugate(v.T)))
        return np.conjugate(eig.T) @ func_H @ eig

def get_spec(H):
    w, v = np.linalg.eigh(H)
    sort_args = np.argsort(w)
    w = w[sort_args]
    v = v[:, sort_args]
    return w, v

def cond_number_sparse(A):
    norm_A = norm(A, 2)
    norm_inv_A = norm(inv(A), 2)
    # print("Norm A: ", norm_A)
    # print("Norm A^-1: ", norm_inv_A)
    return norm_A * norm_inv_A

def compute_ilu_preconditioner(H):
    """
    Computes the ILU preconditioner of a sparse matrix H.

    Parameters:
    H: scipy.sparse matrix
        The input sparse matrix for which to compute the ILU preconditioner.

    Returns:
    M: scipy.sparse.linalg.LinearOperator
        The preconditioner as a linear operator which can be passed directly to GMRES.
    """
    # Convert to CSC format for the spilu function
    H_csc = csc_matrix(H)
    # Perform ILU decomposition
    ilu = spilu(H_csc)

    # Create a linear operator from the ILU decomposition
    Mx = lambda x: ilu.solve(x)
    M = LinearOperator(H.shape, Mx)
    return M

# Below copied from online
# Use pylanczos instead
def lanczos(H,vg, N):
    Lv=np.zeros((N,len(vg)), dtype=complex) # Creates matrix for Lanczos vectors
    Hk=np.zeros((N,N), dtype=complex) #Creates matrix for the Hamiltonian in Krylov subspace
    Lv[0]=vg/np.linalg.norm(vg) #Creates the first Lanczos vector as the normalized guess vector vg

    #Performs the first iteration step of the Lanczos algorithm
    w=np.dot(H,Lv[0])
    a=np.dot(np.conj(w),Lv[0])
    w=w-a*Lv[0]
    Hk[0,0]=a

    #Performs the iterative steps of the Lanczos algorithm
    for j in range(1,N):
        b=(np.dot(np.conj(w),np.transpose(w)))**0.5
        Lv[j]=w/b

        w=np.dot(H,Lv[j])
        a=np.dot(np.conj(w),Lv[j])
        w=w-a*Lv[j]-b*Lv[j-1]

        #Creates tridiagonal matrix Hk using a and b values
        Hk[j,j]=a
        Hk[j-1,j]=b
        Hk[j,j-1]=np.conj(b)

    return (Hk,Lv)


class Eigenvector(np.ndarray):
    def __new__(cls, input_arr):
        if isinstance(input_arr, tuple) and all(isinstance(x, int) for x in input_arr):
            obj = np.zeros(input_arr[0]).view(cls)
            obj[input_arr[1]] = 1.0
        elif isinstance(input_arr, tuple) and all(isinstance(x, np.ndarray) for x in input_arr):
            obj = np.zeros(len(input_arr[1])).view(cls)
            for i, weight in enumerate(input_arr[1]):
                obj += np.sqrt(weight) * (input_arr[0])[:, i]
        else:
            obj = np.asarray(input_arr).view(cls)
        obj = obj / np.sqrt( np.sum( obj ** 2 ) )
        return obj

    def overlap_sq(self, v2):
        return np.abs(np.vdot(self, v2)) ** 2


class resolvent_estim():

    def __init__(self, H, v_0, H_0=None, int_method="gauss", n_int=8, **kwargs):
        self.H = sparse.csr_matrix(H) # Ensures that H is a sparse operator
        self.v_0 = v_0 # Must not be a sparse vector for GMRES
        self.dims = len(v_0)
        self.H_0 = H_0 
        self.exp_0 = exp_val(H, v_0)
        self.exp_sq_0 = exp_val_sq(H, v_0)
        self.var_0 = self.exp_sq_0 - self.exp_0 ** 2
        self.gmres_max_iter = kwargs.get("gmres_max_iter", 20)
        self.gmres_precision = kwargs.get("gmres_precision", 1e-3)
        self.dyson_precision = kwargs.get("dyson_precision", 1e-4)
        self.dyson_max_iter = kwargs.get("dyson_max_iter", 10) # Need to increase after debug completed!!!
        self.x_guess = kwargs.get('x_guess', None)
        self.gen_precond_method = kwargs.get("gen_precond_method", False)
        self.H_lanczos = None

        self.n_int = n_int
        self.int_method = int_method
        self.num_evaluations_GMRES = 0
        self.num_evaluations_BiCGSTAB = 0
        self.num_evaluations_Dyson = 0
        self.H_dot_count = 0
        self.H_dot_count_Lanczos = 0
        self.H_dot_count_Dyson = 0
        self.x_output = []

    def exact_resolvent(self, z):
        def _G(H):
            return 1./(H - z)
        return general_func_exp_val(self.H, self.v_0, _G)

    def lanczos_resolvent(self, z):
        """
        Wrong stuff, DO NOT USE
        """
        def _G(H):
            return 1./(H - z)
        if self.H_lanczos is None:
            self.H_lanczos, _ = lanczos(self.H, self.v_0, self.gmres_max_iter)

        vec = np.zeros(self.gmres_max_iter)
        vec[0] = 1.
        return general_func_exp_val(self.H_lanczos, vec, _G)

    def resummed_dyson_resolvent(self, z, order=2):
        # Need to rewrite this part
        G_0 = sparse.diags(1./(self.H_0.diagonal() - z))
        V = self.H - self.H_0
        diag_val = exp_val(G_0, self.v_0)
        minus_vg0 = - exp_val( np.dot(V, G_0), self.v_0)
        total = 0.0
        for n in range(order+1):
            total += diag_val * ( minus_vg0 ** n)
        return total

    def truncated_dyson_resolvent(self, z, sparse_vec=False):
        def dyson_residue(I_tilde, phi_out, v_0_col, sparse_vec):
            if sparse_vec:
                return sparse.linalg.norm(I_tilde(phi_out) - v_0_col)
            else:
                return np.linalg.norm(I_tilde(phi_out) - v_0_col)
        

        
        if isinstance(self.gen_precond_method, str):
            print(f"Generating preconditioner given method instruction: {self.gen_precond_method}.")
            if self.gen_precond_method == "ILU":
                G_0 = compute_ilu_preconditioner(self.H - z * sparse.identity(self.dims))
        elif self.H_0 == None:
            print("H0 not provided for Truncated Dyson method, divergence may occur using Jacobi H0!")
            G_0 = sparse.diags(1./(sparse.diags(self.H.diagonal()) - z)) #11/2: this line needs to be changed
        else:
            if issparse(self.H_0):
                print("Using provided H0+V splitting for Dyson.")
                H0_minus_zI = self.H_0 - z * sparse.identity(self.dims)
            else:
                print("Given non-sparse H0 splitting for Dyson, converting to sparse...")
                H0_minus_zI = csr_matrix(self.H_0 - z * np.identity(self.dims))
            G_0 = inv(H0_minus_zI) # For future: get H_0 from Ignacio's code as Fermionic operator
        if isinstance(G_0, LinearOperator):
            I_tilde = lambda v: (self.H - z*sparse.identity(self.dims)) @ G_0(v)
        else:
            I_tilde = lambda v: (self.H - z*sparse.identity(self.dims)) @ G_0 @ v # A first-order approximated identity
        V = self.H - self.H_0
        # perturb V and G_0 until obtains phi_step with a smaller norm

        if sparse_vec: # Abandon using sparse v_0_col. Delete for the future.
            v_0_col = sparse.csc_matrix(self.v_0).T 
        else:
            v_0_col = (self.v_0).T # Transpose to get a "column vector"
        phi_step = v_0_col
        phi_out = v_0_col
        residue = dyson_residue(I_tilde, phi_out, v_0_col, sparse_vec)
        counter = 0
        
        while residue > self.dyson_precision:
            if isinstance(G_0, LinearOperator):
                phi_step = - V @ G_0(phi_step)
            else:
                phi_step = - V @ (G_0 @ phi_step)
            if sparse_vec:
                phi_out += phi_step
            else:
                phi_out = phi_out + phi_step
            residue = dyson_residue(I_tilde, phi_out, v_0_col, sparse_vec)
            # error_vec.append(residue)
            self.num_evaluations_Dyson += 2 # 1 for applying V, 1 for evaluating the residue
            counter += 1
            if counter > self.dyson_max_iter:
                print("Maximum Truncated Dyson Iterations reached, exiting optimization.")
                break
            print(f"Truncated Dyson level {counter}, residue: {residue}")
        print("Truncated Dyson iterations: ", self.num_evaluations_Dyson)
        if sparse_vec:
            if isinstance(G_0, LinearOperator):
                return (self.v_0 @ G_0(phi_out))[0]
            else:
                return (self.v_0 @ G_0 @ phi_out)[0]
        else:
            if isinstance(G_0, LinearOperator):
                return self.v_0 @ G_0(phi_out)
            else:
                return self.v_0 @ G_0 @ phi_out

    
    
    def gmres_resolvent(self, z, index):
        def count_evaluations(x):
            # print(f"Residual norm: {x}")
            self.num_evaluations_GMRES += 1

        def linear_operator(x):
            self.H_dot_count += 1
            # 20231112: I think there should be both a minus sign below, and another one in constructing integration path!!!
            return (self.H @ x - z * x)
        
        def precond_from_OP(P):
            """
            Uses spsolve to generate LinearOperator object as preconditioner from a sparse operator.
            Essentially, providing a recipe to compute P^(-1).x
            """
            M_x = lambda x: spsolve(P, x)
            M = linalg.LinearOperator((self.dims, self.dims), M_x)
            return M
        
        # self.x_guess[index] = None
        # self.H_dot_count = 0

        H_operator = linalg.LinearOperator(self.H.shape, matvec=linear_operator)
        # Currently doing a mixed load/compute routine for preconditioners; would be ideal to merge in future!!!
        if isinstance(self.gen_precond_method, str):
            print(f"Generating preconditioner given method instruction: {self.gen_precond_method}.")
            if self.gen_precond_method == "ILU":
                precond = compute_ilu_preconditioner(self.H - z * sparse.identity(self.dims))
            else:
                print("FATAL: Input method for generating preconditioner is not valid!")
            x, info = gmres(H_operator, self.v_0, x0 = self.x_guess[index], M=precond, tol=self.gmres_precision, callback=count_evaluations, callback_type='pr_norm')
            self.x_guess[index] = x # Update initial guess
        else:
            if self.H_0 is None:
                print("No pre-conditioner given, proceed without H0.")
                print(f"Condition number of (H-zI) at z = {z}: ", cond_number_sparse(self.H- z * sparse.identity(self.dims)))
                # print(f"Initial guess vector for point {index}: ", self.x_guess[index])
                x, info = gmres(H_operator, self.v_0, x0 = self.x_guess[index], tol=self.gmres_precision, callback=count_evaluations)
                self.x_guess[index] = x # Point-wise initial guess
            else:
                if issparse(self.H_0):
                    print("Using provided sparse pre-conditioner for GMRES.")
                    H0_minus_zI = self.H_0 - z * sparse.identity(self.dims) # + 0.48*sparse.identity(self.dims)# 2023/11/03: added to test shifting in preconditioner, remove for future
                    print(f"Condition number of (H0-zI)^(-1).(H-zI) at z = {z}: ", cond_number_sparse(inv(H0_minus_zI) @ (self.H- z * sparse.identity(self.dims))))
                    print("Ratio of non-zero entries: ", non_zero_ratio(H0_minus_zI))
                    # M_diag = self.H_0.diagonal() - z
                else:
                    print("Given non-sparse pre-conditioner for GMRES, converting to sparse...")
                    H0_minus_zI = csr_matrix(self.H_0 - z * np.identity(self.dims))
                    print("Ratio of non-zero entries: ", non_zero_ratio(H0_minus_zI))
                    # M_diag = M_sparse.diagonal()
                # M_inv_data = 1.0 / M_sparse.data
                # M_inv = csr_matrix((M_inv_data, M_sparse.indices, M_sparse.indptr), shape=M_sparse.shape)

                precond = precond_from_OP(H0_minus_zI)
                #M_x = lambda x: M_inv @ x
                #M = linalg.LinearOperator((len(self.v_0), len(self.v_0)), M_x)
                
                # print(f"Initial guess vector for point {index}: ", self.x_guess[index])
                x, info = gmres(H_operator, self.v_0, x0 = self.x_guess[index], M=precond, tol=self.gmres_precision, callback=count_evaluations, callback_type='pr_norm')
                #x, info = gmres(H_operator, self.v_0, M=M_inv, x0 = self.x_guess, tol=1e-1, restart=self.gmres_max_iter, maxiter=1, callback=count_evaluations, callback_type='pr_norm')

                self.x_guess[index] = x # Update initial guess
        self.x_output.append(x)

        print("GMRES iters:", self.num_evaluations_GMRES)
        print("Num. times applying mat-vec multiplication: ", self.H_dot_count)
        print("Exit code: ", info)
        return np.dot(self.v_0, x) # Can slightly improve by using sparse dot product if x is also sparse?
    
    def BiCGSTAB_resolvent(self, z, index): # For future: merge with GMRES and correct Overlap_est.py!!!
        def count_evaluations(x):
            # print(f"Residual norm: {x}")
            self.num_evaluations_BiCGSTAB += 1

        def linear_operator(x):
            self.H_dot_count += 1
            # print(x)
            return self.H @ x - z * x
        
        def precond_from_OP(P):
            """
            Uses spsolve to generate LinearOperator object as preconditioner from a sparse operator.
            Essentially, providing a recipe to compute P^(-1).x
            """
            M_x = lambda x: spsolve(P, x)
            M = linalg.LinearOperator((self.dims, self.dims), M_x)
            return M
        
        # self.x_guess[index] = None
        # self.H_dot_count = 0

        H_operator = linalg.LinearOperator(self.H.shape, matvec=linear_operator)
                # Currently doing a mixed load/compute routine for preconditioners; would be ideal to merge in future!!!
        if isinstance(self.gen_precond_method, str):
            print("Generating preconditioner given method instruction.")
            if self.gen_precond_method == "ILU":
                precond = compute_ilu_preconditioner(self.H - z * sparse.identity(self.dims))
            else:
                print("FATAL: Input method for generating preconditioner is not valid!")
            x, info = gmres(H_operator, self.v_0, x0 = self.x_guess[index], M=precond, tol=self.gmres_precision, callback=count_evaluations, callback_type='pr_norm')
            self.x_guess[index] = x # Update initial guess
        else:
            if self.H_0 is None:
                print("No pre-conditioner given, proceed without H0.")
                print(f"Condition number of (H-zI) at z = {z}: ", cond_number_sparse(self.H- z * sparse.identity(self.dims)))
                # print(f"Initial guess vector for point {index}: ", self.x_guess[index])
                x, info = bicgstab(H_operator, self.v_0, x0 = self.x_guess[index], tol=self.gmres_precision, callback=count_evaluations)
                self.x_guess[index] = x # Point-wise initial guess
            else:
                if issparse(self.H_0):
                    print("Using provided sparse pre-conditioner for BiCGSTAB.")
                    H0_minus_zI = self.H_0 - z * sparse.identity(self.dims) # + 0.48*sparse.identity(self.dims)# 2023/11/03: added to test shifting in preconditioner, remove for future
                    print(f"Condition number of (H0-zI)^(-1).(H-zI) at z = {z}: ", cond_number_sparse(inv(H0_minus_zI) @ (self.H- z * sparse.identity(self.dims))))
                    print("Ratio of non-zero entries: ", non_zero_ratio(H0_minus_zI))
                    # M_diag = self.H_0.diagonal() - z
                else:
                    print("Given non-sparse pre-conditioner for BiCGSTAB, converting to sparse...")
                    H0_minus_zI = csr_matrix(self.H_0 - z * np.identity(self.dims))
                    print("Ratio of non-zero entries: ", non_zero_ratio(H0_minus_zI))

                precond = precond_from_OP(H0_minus_zI)
                
                # print(f"Initial guess vector for point {index}: ", self.x_guess[index])
                x, info = bicgstab(H_operator, self.v_0, x0 = self.x_guess[index], M=precond, tol=self.gmres_precision, callback=count_evaluations)
                #x, info = gmres(H_operator, self.v_0, M=M_inv, x0 = self.x_guess, tol=1e-1, restart=self.gmres_max_iter, maxiter=1, callback=count_evaluations, callback_type='pr_norm')

                self.x_guess[index] = x # Update initial guess
        self.x_output.append(x)

        print("BiCGSTAB iters:", self.num_evaluations_BiCGSTAB)
        print("Num. times applying mat-vec multiplication: ", self.H_dot_count)
        print("Exit code: ", info)
        return np.dot(self.v_0, x) # Can slightly improve by using sparse dot product if x is also sparse?

    def overlap_estim(self, **kwargs):
        resolvent = kwargs.get("resolvent", self.exact_resolvent) # Resolvent
        center = kwargs.get("center", self.exp_0) # Center for path
        radius = kwargs.get("radius", 1e-2) # Radius for path
        nint = kwargs.get("nint", self.n_int) # Number of points for integration
        int_method = kwargs.get("int_method", self.int_method) # Integration method
        return np.real_if_close(numerical_contour_integration(resolvent, center, radius, nint, int_method) / (2 * np.pi * 1j))

if __name__ == "__main__":
    A = [[2,3,0],[0,3,0],[2,0,1.5]]
    V = np.array([0,-1,1])
    AS = sparse.csr_matrix(A)
    VS = sparse.csr_matrix(V)
    BS = 1/2*(AS + AS.T)
    a = resolvent_estim(BS,V)
    a.truncated_dyson_resolvent(1.9)