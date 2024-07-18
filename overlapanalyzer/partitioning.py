import time
from juliacall import Main as jl
jl.seval('import Pkg')
jl.seval("using QuantumMAMBO")
mambo = jl.QuantumMAMBO

from openfermion import (
    load_operator,
    save_operator,
    get_sparse_operator,
    jordan_wigner
)
from .read_ham import *



def generateH0(H_F_OP, filename, fileDir, method, cnst, shifts):
    # H0 = {}
    # shifts = {}
    # t0 = time.time()
    H0_F_OP = mambo.MF_planted(H_F_OP, method=method)
    # tf = time.time()
    # print(method+f" took {tf-t0} seconds to run.")
    # runtime[method] = tf-t0

    H_0_OF = mambo.to_OF(H0_F_OP)
    # H_0_OF.terms[()] = cnst # Set the constant term in the calculated H_0 to cnst
    H_0_Q = jordan_wigner(H_0_OF)
    H_0_Q.compress()
    for shift in shifts:
        H_0_Q.terms[()] = cnst + shift
        save_operator(H_0_Q, file_name=f'{filename}_{method}_shift_{shift}.data', data_directory=os.path.join(fileDir,'ham_frag'), allow_overwrite=True, plain_text=True)
    # info = quick_load(path=dir+'/ham_dressed', name=filename[:-5] + ".pkl")
    # H_0_sparse = get_sparse_operator(H_0_OF)
    # shift = info["Energy"][0].item() -  eigsh(H_0_sparse, k=1, which='SA')[0][0] # For future: get this number directly from H_mambo?
    # shifts[method] = shift
    # H_0_sparse += shift*identity(H_0_sparse.shape[0])
    # H0[method] = H_0_OF

def gen_partitions(fileDir, method_list = ['DF-boost'], shift_list = [0]):
    """
    Generate the H0 partitions for the given Hamiltonian.

    Args:
    fileDir (str): The directory containing the Hamiltonian files.
    method_list (list): The list of methods to use for generating the partitions. Can be any combination of ['DF', 'DF-boost', 'CSA'].
    """
    filenames = find_files(os.path.join(fileDir,'ham_fer'),".data")
    ensure_directory_exists(os.path.join(fileDir,'ham_frag'))

    for filename in filenames:
        H_Ferm = load_operator(data_directory=os.path.join(fileDir, 'ham_fer'), file_name=filename, plain_text=True)
        H_Qubit = load_operator(data_directory=os.path.join(fileDir, 'ham_qubit'), file_name=filename, plain_text=True)
        cnst = H_Qubit.terms[()]
        # info = quick_load(path=os.path.join(ham_directory,'iQCC'), name=filename + "_iQCC.pkl")
        H_F_OP = mambo.from_OF(H_Ferm)

        for method in method_list:
            generateH0(H_F_OP, filename[:-5], fileDir, method, cnst, shift_list)



if __name__ == "__main__":

    fileDir = 'hamiltonian_useful/n2'
    gen_partitions(fileDir)

 
