import os
import sys
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator
from openfermion import (
    QubitOperator,
    load_operator,
    get_sparse_operator,
    jw_hartree_fock_state,
    jw_configuration_state
)
import numpy as np
import csv
from overlapanalyzer.read_ham import find_files, quick_load, load_mol_info
from overlapanalyzer.alg_LinearOp import linear_solver, lanczos_lowest_eigenvalue, davidson_lowest_eigenvalue, shifted_lanczos, get_diag_part_QubitOperator
from overlapanalyzer.contour_integration import get_contour, sum_gauss_points, get_contour_with_eigsh_info
from overlapanalyzer.iQCC import apply_iQCC_gens_to_state
from overlapanalyzer.eigen import evals_no_degen, find_subspace_indices, vecs_in_subspace, overlap_with_ON_vecs

def getContourData(v_test, v_hf, eigsh_info, idx, num_points):
    w = eigsh_info['eigsh_energies']
    degen = eigsh_info['degen']
    w_no_degen = evals_no_degen(w, degen)
    exact_eval = w_no_degen[idx]
    start, end = find_subspace_indices(degen, idx)
    degen_vecs = vecs_in_subspace(eigsh_info['eigen_states'], degen, idx)
    ovlp_test = overlap_with_ON_vecs(v_test, degen_vecs)
    ovlp_hf = overlap_with_ON_vecs(v_hf, degen_vecs)
    low, high, contour = get_contour_with_eigsh_info(w, degen, idx, num_points)
    return start, end, ovlp_test, ovlp_hf, low, high, contour, exact_eval


def run_cost_comparison(data_dir = os.path.join('hamiltonian_gen', 'n2'),state_to_eval = 'iQCC'):
    if len(sys.argv) != 5:
        print("Warning: full usage should be: python do_cost_comparison.py <data_dir> <spin> <state_to_eval> <idx>. Using default values for now.")
        data_dir = os.path.join('hamiltonian_gen', 'n2')
        spin = 's0'
        state_to_eval = 'iQCC'
        idx = 0
    else:
        data_dir = sys.argv[1]
        spin = sys.argv[2]
        state_to_eval = sys.argv[3]
        idx = int(sys.argv[4])
    
    current_dir = os.path.dirname(__file__)
    ham_directory = os.path.join(current_dir, data_dir)
    num_electron, num_spin_orb = load_mol_info(os.path.join(ham_directory,'info.csv'))
    n_qubits = num_spin_orb
    original_hams = find_files(os.path.join(ham_directory,'ham_qubit'),spin+".data")


    for i, filename in enumerate(original_hams):
        print("Running cost comparison for: ", filename)
        # spin information is the two characters immediately before the ".data" extension
        spin = filename[-7:-5]
        # phi = quick_load(dir+'/ham_dressed',filename[:-5] + ".pkl")['out_state'] # a numpy ndarray
        H_QubitOperator = load_operator(data_directory=os.path.join(ham_directory,'ham_qubit'), file_name=filename, plain_text=True)
        H_diag_QubitOperator = get_diag_part_QubitOperator(H_QubitOperator)
        H_sparse = get_sparse_operator(H_QubitOperator)
        H_linear = LinearOperator(H_sparse.shape, matvec=lambda x: H_sparse @ x)
        # Build initial states for iterative solvers
        if spin == 't1':
            print("Triplet 2-alpha state")
            phi_ini = jw_configuration_state([i for i in range(num_electron-1)] + [num_electron], n_qubits) # 2-alpha state
        elif spin == 's0':
            print("Singlet state")
            phi_ini = jw_configuration_state([i for i in range(num_electron)], n_qubits)

        # E0, psi_0 = quick_load(dir+'/ham_fer',filename[:-5] + "_Exact_GS_vec.pkl")

        if state_to_eval == 'hf':
            phi = phi_ini
        elif state_to_eval == 'iQCC':
            # Load iQCC results
            iQCC_results = quick_load(os.path.join(ham_directory,'iQCC'),filename[:-5] + "_iQCC.pkl")
            # iQCC_results['num_qubits'] = num_spin_orb
            iQCC_energy = iQCC_results['iQCC_energies'][-1][0]
            # Get iQCC state
            phi = apply_iQCC_gens_to_state(iQCC_results, phi_ini)
        
        # w, v = eigsh(H_sparse, k=2, which='SA', v0 = phi_ini)
        # E0 = w[0]
        # psi_0 = v[:, 0]
        eigsh_results = quick_load(os.path.join(ham_directory,'eigsh'),filename[:-5] + "_eigsh.pkl")
        start_idx, end_idx, overlap_exact, overlap_hf, lb, ub, contour, exact_eval = getContourData(phi, phi_ini, eigsh_results, idx, 8)
        print("Exact phi-ground state overlap: ", overlap_exact)
        print("Exact HF-ground state overlap: ", overlap_hf)

        # Calculate lowest eigenvalue using Lanczos
        output = lanczos_lowest_eigenvalue(H_linear, phi, lb, ub, tol=1e-10, max_iter = 200)
        overlaps = []

        for vector_list in output['eigenvectors']:
            overlaps.append(overlap_with_ON_vecs(phi, vector_list))
        # Prepare zip of method, overlap, matvec_list, vecvec_list, save to csv
        overlap_error = (overlap_exact - np.array(overlaps))/overlap_exact
        # energy_error = np.abs(exact_eval - output['eigenvalues'])
        zipped = zip(output['method'], len(output['method'])*[overlap_exact], overlap_error, output['matvec_list'], output['vecvec_list'], output['runtime_list'])
        header = ["Method", "Overlap_exact", "Overlap_error_relative", "Matvec_count", "Vecvec_count", "Tot_runtime"]
        with open(os.path.join(ham_directory,filename[:-5]+'_costs_'+state_to_eval+'.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(zipped)
        print("Lanczos completed.")

        # Calculate lowest eigenvalue using Davidson
        output = davidson_lowest_eigenvalue(H_linear, phi, H_diag_QubitOperator, n_qubits, lb, ub, tol=1e-10, max_iter = 500)
        overlaps = []
        # for vector in output['eigenvectors']:
        #     overlaps.append(np.abs(np.vdot(vector, phi)) ** 2)
        for vector_list in output['eigenvectors']:
            overlaps.append(overlap_with_ON_vecs(phi, vector_list))
        overlap_error = (overlap_exact - np.array(overlaps))/overlap_exact
        # energy_error = np.abs(E0 - output['eigenvalues'])
        zipped = zip(output['method'], len(output['method'])*[overlap_exact], overlap_error, output['matvec_list'], output['vecvec_list'], output['runtime_list'])
        with open(os.path.join(ham_directory,filename[:-5]+'_costs_'+state_to_eval+'.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(zipped)
        print("Davidson completed.")

        # Calculate overlap using shifted Lanczos
        # contour = get_contour(w[0],(w[1]-w[0])/2, 8)
        output = shifted_lanczos(H_linear, phi, contour["Points"], max_iter = 50)
        overlaps = []
        energy_error = []
        for computed_QF in output['expectation_values']:
            overlaps.append(sum_gauss_points(contour, computed_QF))
            # energy_error.append('NA')
        overlap_error = (overlap_exact - np.array(overlaps))/overlap_exact
        zipped = zip(output['method'], len(output['method'])*[overlap_exact], overlap_error, output['matvec_list'], output['vecvec_list'], output['runtime_list'])
        with open(os.path.join(ham_directory,filename[:-5]+'_costs_'+state_to_eval+'.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(zipped)
        print("Shift-Lanczos completed.")

        # Calculate overlap using gmres
        # Note the need to reshape input arrays
        # psi_0 = psi_0.reshape(-1)
        phi = phi.reshape(-1)
        if state_to_eval == 'iQCC':
            E_state = iQCC_energy
        else:
            E_state = np.vdot(phi, H_sparse @ phi)

        matvec_count = 0
        # matvec_list = []
        computed_QF = []
        # First point
        z = contour["Points"][0]

        A_sparse = diags(z*np.ones(H_sparse.shape[0])) - H_sparse
        M_diag = z * np.ones(H_sparse.shape[0]) - H_sparse.diagonal()
        # M_sparse = diags(M_diag)
        M_inv_diag = 1/M_diag
        M_inv_sparse = diags(M_inv_diag)
        solver = linear_solver((A_sparse @ M_inv_sparse), phi, phi)
        y_est_1, n_matvec = solver.gmres_solve(tol = 1e-2, restart=30) # A low-precision solver for each iteration
        matvec_count += n_matvec
        computed_QF.append(np.vdot(phi, M_inv_sparse @ y_est_1))
        for i in range(1, len(contour["Points"])):
            z = contour["Points"][i]
            A_sparse = diags(z*np.ones(H_sparse.shape[0])) - H_sparse
            M_diag = z * np.ones(H_sparse.shape[0]) - H_sparse.diagonal()
            # M_sparse = diags(M_diag)
            M_inv_diag = 1/M_diag
            M_inv_sparse = diags(M_inv_diag)
            solver = linear_solver((A_sparse @ M_inv_sparse), phi, (contour["Points"][i-1] - E_state)/(z-E_state) * y_est_1)
            y_est, n_matvec = solver.gmres_solve(tol = 1e-2, restart=30) 
            matvec_count += n_matvec
            computed_QF.append(np.vdot(phi, M_inv_sparse @ y_est))
        overlaps = sum_gauss_points(contour, computed_QF)
        overlap_error = (overlap_exact - np.array(overlaps))/overlap_exact
        result = ["GMRES", overlap_exact, overlap_error, matvec_count, "NA"]
        print(result)
        with open(os.path.join(ham_directory,filename[:-5]+'_costs_'+state_to_eval+'.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result)
        print("GMRES completed.")

if __name__ == "__main__":
    # current_dir = os.path.dirname(__file__)
    # test_get_diag_part_from_file('h2o_1.0.data', os.path.join(current_dir, 'gen_hams', 'ham_qubit'))
    run_cost_comparison()