import pickle
import json
import math
import numpy as np
from scipy.sparse.linalg import eigsh
from openfermion import (
    load_operator,
    jw_configuration_state,
    # get_sparse_operator
    get_number_preserving_sparse_operator
)
from overlapanalyzer.eigen import non_degenerate_values, get_exp_val_symmetries, sort_eigen, count_degeneracies, overlap_with_vectors, orthonormalize
from overlapanalyzer.read_ham import *
from overlapanalyzer.utils import exp_val_higher_moment


def calc_eigen(fileDir, spin, num_eigs=10, use_eigsh=False):
    # for spin in  ('s0', 't1'):
    n_elec, n_spin_orb = load_mol_info(os.path.join(fileDir,'info.csv'))
    # if spin == 's0':
    #     # occ_list = hf_occupation_list(n_elec, 0)
    #     hf_state = jw_configuration_state([i for i in range(n_elec)], n_spin_orb)
    # elif spin == 't1':
    #     hf_state = jw_configuration_state([i for i in range(n_elec-1)] + [n_elec], n_spin_orb)


    original_hams = find_files(os.path.join(fileDir,'ham_fer'), spin + ".data") # list of filenames ending with ".data"


    for i, filename in enumerate(original_hams):
        print("Calculating eigenstates for: ", filename)
        H_loaded = load_operator(data_directory=os.path.join(fileDir,'ham_fer'), file_name=filename, plain_text=True)
        print("Hamiltonian loaded.")
        # r = geometry[i]
        H_sparse = get_number_preserving_sparse_operator(H_loaded, n_spin_orb, n_elec)
        print("Hamiltonian converted to sparse.")
        if use_eigsh:
            emax, _ = eigsh(H_sparse, k=1, which='LA')
            w, v = eigsh(H_sparse, k=num_eigs, which='SA')
            w_sorted, v_sorted = sort_eigen(w, v)
        else:
            H_dense = H_sparse.todense()
            w, v = np.linalg.eigh(H_dense)
            w_sorted, v_sorted = sort_eigen(w, v)
            emax = w_sorted[-1]
        degens = count_degeneracies(w_sorted)
        # Orthonormalize the eigenvectors within each degenerate subspace
        v_ON = orthonormalize(v_sorted, degens)
        print("Eigenstates computed.")
        # exp_sz, exp_n, exp_s2 = get_exp_val_symmetries(v_ON, n_spin_orb)
        print("Symmetries computed.")
        hf_state = np.zeros((math.comb(n_spin_orb, n_elec), 1))
        hf_state[0,0] = 1 # Junan 20240928: seems to work but DOUBLE CHECK!!! Also need to generalize for spin
        hf_state_moments = exp_val_higher_moment(H_sparse, hf_state, 40, return_all=True)
        overlaps_with_lowest = overlap_with_vectors(hf_state, v_ON, degens)
        print("Exact HF overlap computed.")
        
        ensure_directory_exists(os.path.join(fileDir, 'eigsh'))
        # save_operator(myiQCC.ham,file_name=filename, data_directory=dir+'/ham_dressed', allow_overwrite=True, plain_text=True)
        # quick_save({"Energy": myiQCC.energies_state_specific, "num_iterations": num_iterations, "num_generators": num_generators}, name=filename[:-5] + ".pkl", path=dir+'/ham_dressed')
        saving_dict = prepare_dict_to_save(eigsh_energies = w_sorted.tolist(),
                                        eigsh_energy_max = emax[0],
                                        degen = degens.tolist(),
                                        eigen_states = v_ON,
                                        overlaps = overlaps_with_lowest, 
                                        hf_state_moments = hf_state_moments.tolist(),
                                        # exp_sz = exp_sz,
                                        # exp_n = exp_n,
                                        # exp_s2 = exp_s2,
                                        molecule=filename[:-5])
        with open(os.path.join(fileDir, 'eigsh', filename[:-5] + "_eigsh.pkl"), 'wb') as f:
            pickle.dump(saving_dict, f)
        # Remove the eigen_states from saving_dict, and save another copy of saving_dict using json
        saving_dict.pop('eigen_states')
        with open(os.path.join(fileDir, 'eigsh', filename[:-5] + "_eigsh.json"), 'w') as f:
            json.dump(saving_dict, f)