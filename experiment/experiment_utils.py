import otmol as otm
from openbabel import pybel   
import time
import pandas as pd
import os
from typing import List
import gc
import psutil

def wc_experiment(mol_pair, 
               data_path: str = None,
               method: str = 'emd',
               n_atoms: int = 3,
               reg: float = 1e-2,
               numItermax: int = 10,
               molecule_cluster_options: str = 'center',
               dataset_name: str = None, # ArbAlignDataWC, 1st2nd, Largest_RMSD
               save: bool = True # whether to save the results
               ):
    results = []
    # Load the molecule pairs from the specified file
    for nameA, nameB in mol_pair:
        start_time = time.time()

        molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA)))
        molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB)))
        X_A, T_A, _ = otm.tl.process_molecule(molA) 
        X_B, T_B, _ = otm.tl.process_molecule(molB)
        optimal_assignment, rmsd_best = otm.tl.cluster_alignment(X_A, X_B, T_A, T_B, case = 'molecule cluster', method = method, n_atoms = n_atoms, reg = reg, numItermax = numItermax, molecule_cluster_options = molecule_cluster_options)
        
        end_time = time.time()

        if not otm.tl.is_permutation(T_A, T_B, optimal_assignment, 'molecule cluster', n_atoms = n_atoms):
            print(nameA, nameB, 'Warning: the assignment is not a water cluster permutation')

        results.append({
            'nameA': nameA,
            'nameB': nameB,
            'method': method,
            'RMSD(OTMol)': rmsd_best,
            '# atoms': X_A.shape[0],
            'time': end_time - start_time,
            'assignment': optimal_assignment,
        }) 
        print(nameA, nameB, method, f"{rmsd_best:.2f}", f"{end_time - start_time:.2f}s")
    
    results_df = pd.DataFrame(results)
    if save == True:
        results_df.to_csv(os.path.join('./otmolOutput', f'wc_results_{dataset_name}_{method}.csv'), index=False)
    return results_df


def ng_experiment(mol_pair, 
               data_path: str = None,
               p_list: list = None,
               method: str = 'emd',
               reg: float = 1e-4,
               numItermax: int = 10000,
               save: bool = True # whether to save the results
               ):
    results = []
    # Load the molecule pairs from the specified file
    for nameA, nameB in mol_pair:
        start_time = time.time()

        molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
        molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
        X_A, _, _ = otm.tl.process_molecule(molA) 
        X_B, _, _ = otm.tl.process_molecule(molB)
        optimal_assignment, rmsd_best, p_best = otm.tl.cluster_alignment(X_A = X_A, X_B = X_B, case = 'same element', method = method, p_list = p_list, reg = reg, numItermax = numItermax)
        
        end_time = time.time()

        if not otm.tl.is_permutation(perm=optimal_assignment):
            print(nameA, nameB, 'Warning: the assignment is not 1 to 1')

        results.append({
            'nameA': nameA,
            'nameB': nameB,
            'method': method,
            'RMSD(OTMol)': rmsd_best,
            '# atoms': X_A.shape[0],
            'time': end_time - start_time,
            'p': p_best,
            'assignment': optimal_assignment,
        }) 
        print(nameA, nameB, method, f"{rmsd_best:.2f}", f"{end_time - start_time:.2f}s")
    
    results_df = pd.DataFrame(results)
    if save == True:
        results_df.to_csv(os.path.join('./otmolOutput', f'ng_results_{method}.csv'), index=False)
    return results_df


def experiment(
        data_path: str = None,
        mol_pair: list = None, 
        setup: str = 'element name',
        method: list= ['fGW', 'emd'], 
        alpha_list: list = None,
        molecule_sizes: List[int] = None,
        reg: float = 1e-2,
        dataset_name: str = None, # FGG, S1, cyclic_peptides
        save: bool = False # whether to save the results
        ):
    results = []
    # Load the molecule pairs from the specified file
    for nameA, nameB in mol_pair:
        if setup == 'element name':
            molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
            molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
            X_A, T_A, B_A = otm.tl.process_molecule(molA) 
            X_B, T_B, B_B = otm.tl.process_molecule(molB)
        if setup == 'atom type':
            if dataset_name == 'S1':
                X_A, T_A = otm.tl.parse_sy2(os.path.join(data_path, nameA + '_chimera.sy2'))
                X_B, T_B = otm.tl.parse_sy2(os.path.join(data_path, nameB + '_chimera.sy2'))
            else:
                X_A, T_A = otm.tl.parse_sy2(os.path.join(data_path, nameA + '.sy2'))
                X_B, T_B = otm.tl.parse_sy2(os.path.join(data_path, nameB + '.sy2'))
        if setup == 'atom connectivity':
            molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
            molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
            X_A, _, _ = otm.tl.process_molecule(molA) 
            X_B, _, _ = otm.tl.process_molecule(molB)
            T_A = otm.tl.parse_mna(os.path.join(data_path, nameA + '.mna'))
            T_B = otm.tl.parse_mna(os.path.join(data_path, nameB + '.mna'))

        optimal_assignment, rmsd_best, alpha_best = otm.tl.molecule_alignment(X_A, X_B, T_A, T_B, method = method, alpha_list = alpha_list, molecule_sizes = molecule_sizes, reg = reg)
            
        if not otm.tl.is_permutation(T_A = T_A, T_B = T_B, perm = optimal_assignment, case = 'single'): 
            print(nameA, nameB, 'Warning: not a proper permutation')
        results.append({
            'nameA': nameA,
            'nameB': nameB,
            'RMSD(OTMol+{})'.format(setup): rmsd_best,
            '# atoms': X_A.shape[0],
            'alpha': alpha_best,
            'assignment': optimal_assignment,
        }) 
        print(nameA, nameB, f"{rmsd_best:.2f}")

    results_df = pd.DataFrame(results)
    if save == True:
        results_df.to_csv(os.path.join('./otmolOutput', f'{dataset_name}_results_{setup}_{method}.csv'), index=False)

    return pd.DataFrame(results)


def cp_experiment(
        data_path: str = None,
        mol_pair: list = None, 
        method: list = ['fGW', 'emd'], 
        alpha_list: list = None,
        ):
    results = []
    # Load the molecule pairs from the specified file
    for subfolder, nameA, nameB in mol_pair:
        molA = next(pybel.readfile('xyz', os.path.join(data_path, subfolder, nameA)))
        molB = next(pybel.readfile('xyz', os.path.join(data_path, subfolder, nameB)))
        X_A, T_A, B_A = otm.tl.process_molecule(molA) 
        X_B, T_B, B_B = otm.tl.process_molecule(molB)
        
        optimal_assignment, rmsd_best, alpha_best = otm.tl.molecule_alignment(X_A, X_B, T_A, T_B, method = method, alpha_list = alpha_list)

        if not otm.tl.is_permutation(T_A = T_A, T_B = T_B, perm = optimal_assignment, case = 'single'):
            print(nameA, nameB, 'Warning: Not a proper assignment')
        results.append({
            'nameA': nameA,
            'nameB': nameB,
            'RMSD(OTMol)': rmsd_best,
            'alpha': alpha_best,
            'assignment': optimal_assignment,
            }) 
        print(nameA, nameB, f"{rmsd_best:.2f}")

    return pd.DataFrame(results)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024.0**2  # Convert to MB


def profile_memory(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        gc.collect()
        
        # Get initial memory
        start_mem = process.memory_info().rss
        
        result = func(*args, **kwargs)
        
        # Get final memory
        end_mem = process.memory_info().rss
        
        memory_used = (end_mem - start_mem) / 1024.0**2  # Convert to MB
        print(f"Memory usage: {memory_used:.2f} MB")
        return result, memory_used
    return wrapper