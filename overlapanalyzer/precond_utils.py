from krypy.linsys import LinearSystem, Gmres
from krypy.deflation import DeflatedGmres
import unittest
import itertools
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
# import pickle
import csv
import pandas as pd
import ast

import os
import numpy as np
import scipy.sparse.linalg as spla
from openfermion import jw_configuration_state, load_operator, QubitOperator
from overlapanalyzer.alg_LinearOp import get_diag_part_QubitOperator, get_sparse_operator
from overlapanalyzer.read_ham import quick_load, load_mol_info, find_files
from overlapanalyzer.iQCC import apply_iQCC_gens_to_state
from overlapanalyzer.cost_comparison import getContourDataFromEigsh
from overlapanalyzer.contour_integration import sum_gauss_points

class TestPrecond(unittest.TestCase):
    def calculateInnerProducts(self, H_q, phi, zi, initial_state, preconditioner, tol=1e-5):
        print("Solving for z = ", zi)
        linear_system = LinearSystem(get_sparse_operator(QubitOperator('', zi)-H_q), phi, Ml=preconditioner)
        solver = Gmres(linear_system, x0=initial_state, explicit_residual=True, store_all_xk=True, tol=tol)
        xks = solver.xk_all
        outList = [np.inner(phi, xk.flatten()) for xk in xks]
        print("First 5 integrands: ", outList[:5])
        return outList
    # for zi in final_vecs.keys():

def genOverlapErrors(ham_directory, filename, dressed_ham=False):
    num_electron, num_spin_orb = load_mol_info(os.path.join(ham_directory,'info.csv'))
    spin = filename[-2:]

    if spin == 't1':
        print("Triplet 2-alpha state")
        phi_ini = jw_configuration_state([i for i in range(num_electron-1)] + [num_electron], num_spin_orb) # 2-alpha state
    elif spin == 's0':
        print("Singlet state")
        phi_ini = jw_configuration_state([i for i in range(num_electron)], num_spin_orb)
    else:
        raise ValueError("No valid spin state detected from filename!")

    if not dressed_ham:
        H_loaded_q = load_operator(data_directory=os.path.join(os.path.join(ham_directory,'ham_qubit')), file_name=filename+'.data', plain_text=True)
    else:
        H_loaded_q = load_operator(data_directory=os.path.join(os.path.join(ham_directory,'iQCC')), file_name=filename+'_dressed.data', plain_text=True)
    # H_Ferm = reverse_jordan_wigner(H_loaded_q)
    # H_F_OP = mambo.from_OF(H_Ferm)
    H_diag_q = get_diag_part_QubitOperator(H_loaded_q)
    # Load and construct the iQCC output state
    iQCC_results = quick_load(os.path.join(ham_directory,'iQCC'),filename + "_iQCC.pkl")
    # iQCC_results['num_qubits'] = num_spin_orb
    iQCC_energy = iQCC_results['iQCC_energies'][-1][0]
    # Get iQCC state
    eigsh_results = quick_load(os.path.join(ham_directory,'eigsh'),filename + "_eigsh.pkl")
    phi = apply_iQCC_gens_to_state(iQCC_results, phi_ini)
    start_idx, end_idx, overlap_exact, overlap_hf, lb, ub, contour, exact_eval = getContourDataFromEigsh(phi, phi_ini, eigsh_results, 0, 8) # idx=0 for ground state
    if dressed_ham:
        phi = phi_ini # Use HF state if the Hamiltonian is dressed
    
    # Instantiate the test case
    tests = TestPrecond()
    # Define your list of zi values
    zi_values = contour["Points"]
    # Run the tests and collect the results
    results = {}

    H0_CSA = load_operator(data_directory=os.path.join(ham_directory,'ham_frag'), file_name=filename+"_CSA_shift_0.data", plain_text=True)
    H0_DF_boost = load_operator(data_directory=os.path.join(ham_directory,'ham_frag'), file_name=filename+"_DF-boost_shift_0.data", plain_text=True)
    H0_DF = load_operator(data_directory=os.path.join(ham_directory,'ham_frag'), file_name=filename+"_DF_shift_0.data", plain_text=True) 
    H0_list = [H_diag_q, H0_DF_boost, H0_DF, H0_CSA]
    preconditioners_labels = ['1/(z-H_diag)', '1/(z-H_DF_boost)', '1/(z-H_DF)', '1/(z-H_CSA)', 'None']
    for i, zi in enumerate(zi_values):
        # Generate your lists of initial states and preconditioners based on zi
        initial_states = [1/(zi-exact_eval)*phi, None] # Change exact_eval to center of integration path in the future
        initial_states_labels = ['1/(z-c)*|iQCC>', 'All-0']
        preconditioners = [spla.inv(get_sparse_operator(QubitOperator('', zi)-H0)) for H0 in H0_list] + ['None']
        
        for j, initial_state in enumerate(initial_states):
            for k, preconditioner in enumerate(preconditioners):
                # Use a tuple (i, j, k) as the key
                results[(i, j, k)] = tests.calculateInnerProducts(H_loaded_q, phi, zi, initial_state, preconditioner)

    # Process the results
    overlapError = {}
    overlaps = {}
    if dressed_ham:
        overlapFilename = filename+"_overlapError_dressed"
    else:
        overlapFilename = filename+"_overlapError"
    
    for j in range(len(initial_states)):
        for k in range(len(preconditioners)):
            min_length = min(len(results[(i, j, k)]) for i in range(len(zi_values)))
            overlapError[(j, k)] = []
            for l in range(min_length):
                InnerProds_itr_l = [results[(i, j, k)][l] for i in range(len(zi_values))]
                overlap_itr_l = sum_gauss_points(contour, InnerProds_itr_l)
                overlaps[(j,k,l)] = overlap_itr_l
                overlapError[(j, k)].append(overlap_exact - overlap_itr_l)
    # pickle.dump(overlapError, open(os.path.join(ham_directory,overlapFilename+".pkl"), "wb"))
    file_path = os.path.join(ham_directory, overlapFilename+".csv")
    file_exists = os.path.isfile(file_path)
    
    # Open the file in append mode if it exists, otherwise in write mode
    with open(file_path, "a" if file_exists else "w", newline='') as f:
        writer = csv.writer(f)
        # Write the column headers only if the file did not exist before
        if not file_exists:
            writer.writerow(['Initial state', 'Preconditioner', 'Iteration', 'Overlap Error', 'Overlap'])
        for (j, k), error in overlapError.items():
            for l, err in enumerate(error):
                writer.writerow([initial_states_labels[j], preconditioners_labels[k], l, err, overlaps[(j,k,l)]])

def compareShifts(ham_directory, filename):
    num_electron, num_spin_orb = load_mol_info(os.path.join(ham_directory,'info.csv'))
    spin = filename[-2:]

    if spin == 't1':
        print("Triplet 2-alpha state")
        phi_ini = jw_configuration_state([i for i in range(num_electron-1)] + [num_electron], num_spin_orb) # 2-alpha state
    elif spin == 's0':
        print("Singlet state")
        phi_ini = jw_configuration_state([i for i in range(num_electron)], num_spin_orb)
    else:
        raise ValueError("No valid spin state detected from filename!")

    H_loaded_q = load_operator(data_directory=os.path.join(os.path.join(ham_directory,'ham_qubit')), file_name=filename, plain_text=True)
    # Load and construct the iQCC output state
    iQCC_results = quick_load(os.path.join(ham_directory,'iQCC'),filename + "_iQCC.pkl")
    # iQCC_results['num_qubits'] = num_spin_orb
    iQCC_energy = iQCC_results['iQCC_energies'][-1][0]
    # Get iQCC state
    eigsh_results = quick_load(os.path.join(ham_directory,'eigsh'),filename + "_eigsh.pkl")
    phi = apply_iQCC_gens_to_state(iQCC_results, phi_ini)
    start_idx, end_idx, overlap_exact, overlap_hf, lb, ub, contour, exact_eval = getContourDataFromEigsh(phi, phi_ini, eigsh_results, 0, 8) # idx=0 for ground state
    
    filenames = find_files(os.path.join(ham_directory, 'ham_frag'), '.data')
    # Instantiate the test case
    tests = TestPrecond()
    # Define your list of zi values
    zi_values = contour["Points"]
    # Run the tests and collect the results
    results = {}

    # Extract the shift from the filenames. The shift is immediately before the .data extension and immediately after the _
    shifts = [int(filename.split('_')[-1][:-5]) for filename in filenames]
    # Sort both the filenames and shifts according to the values in the shifts list
    shifts, filenames = zip(*sorted(zip(shifts, filenames)))
    for k, name in enumerate(filenames):
        H_0 = load_operator(data_directory=os.path.join(ham_directory,'ham_frag'), file_name=name, plain_text=True)
        
        for i, zi in enumerate(zi_values):
            # Generate your lists of initial states and preconditioners based on zi
            initial_states = [1/(zi-exact_eval)*phi] # Change exact_eval to center of integration path in the future
            initial_states_labels = ['1/(z-c)*|iQCC>']
            preconditioner = spla.inv(get_sparse_operator(QubitOperator('', zi)-H_0))
            # preconditioners_labels = ['1/(z-H_diag)']
            for j, initial_state in enumerate(initial_states):
                # for k, preconditioner in enumerate(preconditioners):
                    # Use a tuple (i, j, k) as the key
                results[(i, j, k)] = tests.calculateInnerProducts(H_loaded_q, phi, zi, initial_state, preconditioner)

    # Process the results
    overlapError = {}
    overlaps = {}
    overlapFilename = filename+"_overlapError_ShiftComp"
    
    for j in range(len(initial_states)):
        for k, shift in enumerate(shifts):
            min_length = min(len(results[(i, j, k)]) for i in range(len(zi_values)))
            overlapError[(j, k)] = []
            for l in range(min_length):
                InnerProds_itr_l = [results[(i, j, k)][l] for i in range(len(zi_values))]
                overlap_itr_l = sum_gauss_points(contour, InnerProds_itr_l)
                overlaps[(j,k,l)] = overlap_itr_l
                overlapError[(j, k)].append(overlap_exact - overlap_itr_l)
    # pickle.dump(overlapError, open(os.path.join(ham_directory,overlapFilename+".pkl"), "wb"))
    # Save the results to a csv file
    file_path = os.path.join(ham_directory, overlapFilename+".csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a" if file_exists else "w", newline='') as f:
        writer = csv.writer(f)
        # Write the column headers only if the file did not exist before
        if not file_exists:
            writer.writerow(['Initial state', 'Shift', 'Iteration', 'Overlap Error', 'Overlap'])
        for (j, k), error in overlapError.items():
            for l, err in enumerate(error):
                writer.writerow([initial_states_labels[j], shifts[k], l, err, overlaps[(j,k,l)]])

def plotOverlapErrors(ham_directory, filename, dressed_ham=False, compareShifts=False, initial_states_subset=None, shifts_subset=None):
    if not compareShifts:
        cases_label = 'Preconditioner'
        if dressed_ham:
            overlapFilename = filename+"_overlapError_dressed"
        else:
            overlapFilename = filename+"_overlapError"
    else:
        cases_label = 'Shift'
        overlapFilename = filename+"_overlapError_ShiftComp"
    # overlapError = pickle.load(open(os.path.join(ham_directory,overlapFilename+".pkl"), "rb"))
    overlapErrors = pd.read_csv(os.path.join(ham_directory, overlapFilename+".csv"))
    # Get the labels from the first entry of the first and second column
    initial_states_labels = overlapErrors['Initial state'].unique() if initial_states_subset is None else initial_states_subset
    shifts = overlapErrors[cases_label].unique() if shifts_subset is None else shifts_subset
    # Plot the results for each initial state and shift, as a function of the index l, from the pandas dataframe
    for j, k in itertools.product(initial_states_labels, shifts):
        SelectedDataFrame = overlapErrors[(overlapErrors['Initial state'] == j) & (overlapErrors[cases_label] == k)]
        label_jk = f'shift={k}' if len(initial_states_labels) == 1 else f'init_state={j}, {cases_label}={k}'
        plt.plot(SelectedDataFrame['Iteration'], SelectedDataFrame['Overlap Error'], label=label_jk)
    plt.xlabel('Iterations')
    plt.ylabel(r'$O_{exact}-O_{approx}$')
    if len(initial_states_labels) == 1:
        plt.title(f'Init state={initial_states_labels[0]}, ' + filename)
    else:
        plt.title(filename + ", overlap error")
    plt.legend()
    # Save the plot as pdf
    plt.savefig(os.path.join(ham_directory,overlapFilename+".pdf"))
    plt.show()


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    ham_directory = os.path.join(current_dir, 'hamiltonian_useful/lih_2.6')
    filename = 'lih_2.6_4_12_s0'
    genOverlapErrors(ham_directory, filename, dressed_ham=True)
    # compareShifts(ham_directory, filename)
