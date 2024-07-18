import pickle
from scipy.sparse.linalg import eigsh
from openfermion import (
    load_operator,
    jw_configuration_state,
    get_sparse_operator
)
from .eigen import hf_occupation_list, get_exp_val_symmetries, sort_eigen, count_degeneracies, overlap_with_vectors, orthonormalize
from .read_ham import *


def calc_eigen(fileDir, num_eigs=10):
    for spin in  ('s0', 't1'):
        n_elec, n_spin_orb = load_mol_info(os.path.join(fileDir,'info.csv'))
        if spin == 's0':
            occ_list = hf_occupation_list(n_elec, 0)
        elif spin == 't1':
            occ_list = hf_occupation_list(n_elec, 2)
        hf_state = jw_configuration_state(occ_list, n_spin_orb)


        original_hams = find_files(os.path.join(fileDir,'ham_qubit'), spin + ".data") # list of filenames ending with ".data"


        for i, filename in enumerate(original_hams):
            print("Calculating eigenstates for: ", filename)
            H_loaded = load_operator(data_directory=os.path.join(fileDir,'ham_qubit'), file_name=filename, plain_text=True)
            print("Hamiltonian loaded.")
            # r = geometry[i]
            H_sparse = get_sparse_operator(H_loaded)
            print("Hamiltonian converted to sparse.")
            emax, _ = eigsh(H_sparse, k=1, which='LA', v0=hf_state)
            w, v = eigsh(H_sparse, k=num_eigs, which='SA', v0=hf_state)
            w_sorted, v_sorted = sort_eigen(w, v)
            degens = count_degeneracies(w_sorted)
            # Orthonormalize the eigenvectors within each degenerate subspace
            v_ON = orthonormalize(v_sorted, degens)
            print("Eigenstates computed.")
            exp_sz, exp_n, exp_s2 = get_exp_val_symmetries(v_ON, n_spin_orb)
            print("Symmetries computed.")
            overlaps_with_lowest = overlap_with_vectors(hf_state, v_ON, degens)
            print("Exact HF overlap computed.")
            
            ensure_directory_exists(os.path.join(fileDir, 'eigsh'))
            # save_operator(myiQCC.ham,file_name=filename, data_directory=dir+'/ham_dressed', allow_overwrite=True, plain_text=True)
            # quick_save({"Energy": myiQCC.energies_state_specific, "num_iterations": num_iterations, "num_generators": num_generators}, name=filename[:-5] + ".pkl", path=dir+'/ham_dressed')
            saving_dict = prepare_dict_to_save(hf_occupation = occ_list,
                                            eigsh_energies = w_sorted,
                                            eigsh_energy_max = emax[0],
                                            degen = degens,
                                            eigen_states = v_ON,
                                            overlaps = overlaps_with_lowest, 
                                            exp_sz = exp_sz,
                                            exp_n = exp_n,
                                            exp_s2 = exp_s2,
                                            molecule=filename)
            with open(os.path.join(fileDir, 'eigsh', filename[:-5] + "_eigsh.pkl"), 'wb') as f:
                pickle.dump(saving_dict, f)
            # quick_save(saving_dict, name=filename[:-5] + f"_{num_generators}_gens_iQCC.pkl", path=ham_directory+'/iQCC_s0')