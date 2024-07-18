
import time
import sys
import numpy as np
from qcc_240202.QCC import iQCC
import pathos.multiprocessing as multiprocessing
from openfermion import (
    load_operator,
    save_operator,
    sz_operator,
    s_squared_operator,
    jordan_wigner,
    get_sparse_operator,
    QubitOperator
)
from .read_ham import *
from .eigen import hf_occupation_list

def apply_iQCC_gens_to_state(iqcc_results, state):
    """
    Apply the iQCC generators to the state.
    Need optimization using SymplecticPauli to avoid the need to convert to sparse matrix.
    """
    n_qubits = iqcc_results['num_qubits']
    id_sparse = get_sparse_operator(QubitOperator(''), n_qubits=n_qubits)
    U_total = id_sparse
    for ampls, gens in zip(iqcc_results['ampls_at_itr'], iqcc_results['gens_at_itr']):
        for ampl, gen in zip(ampls, gens):
            U = np.cos(ampl / 2) * id_sparse - 1j * np.sin(ampl / 2)  * get_sparse_operator(gen, n_qubits=n_qubits)
            U_total = U @ U_total
    return U_total @ state

def run_iQCC(fileDir, spin = 's0', num_iterations = 3, num_generators = 6, has_geometry_param = False):
    print("Starting iQCC calculation for ", fileDir)
    if len(sys.argv) != 4:
        print("Using values passed in as arguments.")
        print(f"Spin: {spin}, num_iterations: {num_iterations}, num_generators: {num_generators}")
        # spin = 's0'
        # num_iterations = 3
        # num_generators = 6
    else:
        print("Using command line arguments.")
        spin = sys.argv[1]
        num_iterations = int(sys.argv[2])
        num_generators = int(sys.argv[3])


    N_CORES = multiprocessing.cpu_count()
    n_elec, n_spin_orb = load_mol_info(os.path.join(fileDir,'info.csv'))
    hf_occupation = hf_occupation_list(n_elec, spin)
    if spin == 't1':
        print("Preparing triplet alpha reference state.")
        refs = (n_elec-1)*(1, ) +(0,1, ) + (n_spin_orb - n_elec-1) *(0, )
        if n_elec % 2 != 0 or n_spin_orb % 2 != 0:
            raise ValueError("Number of electrons and spin orbitals must be even for triplet alpha reference state!")
        sz = sz_operator(int(n_spin_orb // 2))
        s2 = s_squared_operator(int(n_spin_orb // 2))
        W_penalty = jordan_wigner(s2 - 3*sz + 1)
    elif spin == 's0':
        print("Preparing singlet reference state.")
        refs = n_elec*(1, ) + (n_spin_orb - n_elec) *(0, )
        

    original_hams = find_files(os.path.join(fileDir,'ham_qubit'),spin+".data") # list of filenames ending with ".data"
    if has_geometry_param:
        geometry = extract_numbers_from_filenames(original_hams)


    for i, filename in enumerate(original_hams):
        print("Running iQCC calculation for ", filename)
        H_loaded = load_operator(data_directory=fileDir+'/ham_qubit', file_name=filename, plain_text=True)
        if spin == 't1':
            H_op = H_loaded + 0.4 * W_penalty
        elif spin == 's0':
            H_op = H_loaded
        # Do QCC
        myiQCC = iQCC(
        hamiltonian=H_op,  # QubitOperator
        references=refs,
        n_gen=num_generators,
        n_cores=N_CORES,
        )
        ham_terms = []
        t_init = time.time()
        tot_runtime = []
        for _ in range(num_iterations):
            myiQCC.step()
            ham_terms.append(len(myiQCC.ham.terms))
            # ham_terms.append(myiQCC.n_terms_at_itr)
            tot_runtime.append(time.time() - t_init)
            # Write to a file the current gens_at_itr, ampls_at_itr, and energies_at_itr
            quick_save({"gens_at_itr": myiQCC.gens_at_itr[:len(myiQCC.gens_at_itr)][::-1], "ampls_at_itr": myiQCC.ampls_at_itr[:len(myiQCC.ampls_at_itr)][::-1], "iQCC_energies": myiQCC.energies_at_itr, "runtime": tot_runtime}, name=filename[:-5] + f"_{num_generators}_gens_iQCC_temp.pkl", path=fileDir+'/iQCC')
        myiQCC.cpu_pool.terminate()
        print('\nFinished {} iterations after {} seconds'.format(num_iterations, round(time.time() - t_init,4)))

        gens_at_itr = myiQCC.gens_at_itr[:len(myiQCC.gens_at_itr)][::-1]
        ampls_at_itr = myiQCC.ampls_at_itr[:len(myiQCC.ampls_at_itr)][::-1]
        iQCC_energies = myiQCC.energies_at_itr
        ham_terms = myiQCC.n_terms_at_itr
        ensure_directory_exists(os.path.join(fileDir, 'iQCC'))
        # save_operator(myiQCC.ham,file_name=filename, data_directory=dir+'/ham_dressed', allow_overwrite=True, plain_text=True)
        # quick_save({"Energy": myiQCC.energies_state_specific, "num_iterations": num_iterations, "num_generators": num_generators}, name=filename[:-5] + ".pkl", path=dir+'/ham_dressed')
        saving_dict = prepare_dict_to_save(num_qubits = len(refs), 
                                        num_iterations=num_iterations, 
                                        num_generators=num_generators, 
                                        ham_terms = ham_terms,
                                        gens_at_itr=gens_at_itr, 
                                        ampls_at_itr=ampls_at_itr, 
                                        refs=refs, 
                                        hf_occupation = hf_occupation,
                                        iQCC_energies = iQCC_energies, 
                                        runtime = tot_runtime,
                                        molecule=filename)        
        if has_geometry_param:
            saving_dict['geometry'] = geometry[i]
        quick_save(saving_dict, name=filename[:-5] + f"_iQCC.pkl", path=os.path.join(fileDir, 'iQCC'))
        # Save dressed Hamiltonian
        save_operator(myiQCC.ham,file_name=filename[:-5] + f"_dressed.data", data_directory=os.path.join(fileDir, 'iQCC'), allow_overwrite=True, plain_text=True)

# quick_save(f'iQCC generators: {num_generators}\niQCC iterations: {num_iterations}', name='info.txt',path=dir+'/ham_dressed',file_type='text')
if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    ham_directory = os.path.join(current_dir, 'hamiltonian_gen', 'n2')
    # Obtain the number of iterations and generators from the entries following this script's filename
    run_iQCC(ham_directory, 's0')