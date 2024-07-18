import os
import re
import pickle
# from do_partition import H_partition # Uncomment to resolve H_partition objects using quick_load

def quick_save(data, name, path, file_type='general'):
    try:
        os.makedirs(path, exist_ok=True)
        
        # Determine the mode and action based on file_type
        mode, action = 'wb', pickle.dump
        if file_type == 'text':
            mode, action = 'w', lambda d, f: f.write(d)
        elif file_type == 'bin':
            action = lambda d, f: f.write(d)
        elif file_type != 'general':
            raise ValueError("Unsupported file type: {}".format(file_type))

        with open(os.path.join(path, name), mode) as f:
            action(data, f)

    except Exception as e:
        print(f"Error saving the file: {e}")

def quick_load(path, name, file_type='general'):
    try:
        mode, action = 'rb', pickle.load
        if file_type == 'text':
            mode, action = 'r', lambda f: f.read()
        elif file_type == 'bin':
            action = lambda f: f.read()
        elif file_type != 'general':
            raise ValueError("Unsupported file type: {}".format(file_type))

        with open(os.path.join(path, name), mode) as f:
            return action(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {os.path.join(path, name)}") from e
    except Exception as e:
        print(f"Error loading the file: {e}")
        return None

def load_mol_info(path):
    """
    Loads the number of electrons and number of spin orbitals given the directory path.
    """
    import pandas as pd
    df = pd.read_csv(path, header=0)

    # Extract the values
    n_electron = df['N_electron'][0]
    n_spin_orb = df['N_spin_orb'][0]

    return n_electron, n_spin_orb

def read_xyz(xyz_file):
    """
    Read the xyz file and return the coordinates.
    """
    with open(xyz_file, "r") as f:
        lines = f.readlines()
    xyz = "".join(lines[2:])
    return xyz

def find_files(path, extension):
    """
    Returns a list of all file names ending with in the specified path, sorted in lexicographical order.
    
    Parameters:
    - path: The path to search for particular extension files.
    
    Returns:
    - A list of lexicographically sorted file names with particular extension extension in the given path.
    """
    all_files = os.listdir(path)
    
    # Filter out files that don't end with specified extension
    data_files = sorted([file for file in all_files if file.endswith(extension)])
    
    return data_files


def extract_numbers_from_filenames(data_files):
    """
    Extract numbers from filenames like "jw_2.3.data" and "bk_full_0.75.data".
    
    Parameters:
    - data_files: List of file names.
    
    Returns:
    - A list of extracted numbers.
    """    
    pattern = r'(\d+\.\d+)'
    
    numbers = []
    for file in data_files:
        match = re.search(pattern, file)
        if match:
            # Convert the matched substring to a float and append to the list
            numbers.append(float(match.group(1)))
    
    return numbers

def extract_common_prefix(data_files):
    """
    Scans through the list of filenames to see if they all start with a common word.
    If they do, returns that word. Otherwise, raises an error.

    Parameters:
    - data_files: List of file names.

    Returns:
    - The common word (prefix) if it exists.
    
    Raises:
    - ValueError: If the files don't have a common prefix.
    """
    
    # Extract prefixes using list comprehension
    prefixes = [file.split('_')[0] for file in data_files]
    
    # Check if all prefixes are the same
    if len(set(prefixes)) == 1:
        return prefixes[0]
    else:
        raise ValueError("The provided files don't have a common prefix!")


def ensure_directory_exists(dir_path):
    """
    Checks if a directory exists at the specified path. If not, creates it.
    
    Parameters:
    - dir_path: Path to the directory to check or create.
    
    Returns:
    - True if the directory was created, False if it already existed.
    """
    
    # Check if directory exists
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True
    return False


def prepare_dict_to_save(**kwargs):
    my_dict = {}
    merged_dict = my_dict | kwargs
    return merged_dict

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    ham_directory = os.path.join(current_dir, 'hamiltonian_gen_test', 'beh2')
    n_elec, n_spin_orb = load_mol_info(ham_directory+'/info.csv')
    print(n_elec, n_spin_orb)
