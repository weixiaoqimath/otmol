import otmol as otm
from openbabel import pybel   
import time
import pandas as pd
import os
from typing import List
import gc
import psutil
import numpy as np
from pathlib import Path



def wc_experiment(mol_pair, 
               data_path: str = None,
               method: str = 'emd',
               n_atoms: int = 3,
               reg: float = 1e-2,
               numItermax: int = 10,
               representative_option: str = 'center',
               dataset_name: str = None, # ArbAlignDataWC, 1st2nd, Largest_RMSD
               save: bool = False, # whether to save the results
               n_trials: int = 300,
               ):
    results = []
    # Load the molecule pairs from the specified file
    for nameA, nameB in mol_pair:
        start_time = time.time()

        molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA)))
        molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB)))
        X_A, T_A, _ = otm.tl.process_molecule(molA) 
        X_B, T_B, _ = otm.tl.process_molecule(molB)
        optimal_assignment, rmsd_best = otm.tl.cluster_alignment(
            X_A, X_B, T_A, T_B, case = 'molecule cluster', 
            method = method, n_atoms = n_atoms, 
            reg = reg, numItermax = numItermax, 
            representative_option = representative_option,
            n_trials = n_trials,
            save_path = f'./otmol_output/{nameB.split(".")[0]}_to_{nameA.split(".")[0]}_otmol_rep={representative_option}.xyz'
            )
            
        end_time = time.time()

        if not otm.tl.is_permutation(T_A, T_B, optimal_assignment, 'molecule cluster', n_atoms = n_atoms):
            print(nameA, nameB, 'Warning: the assignment is not a water cluster permutation')

        results.append({
                'nameA': nameA,
                'nameB': nameB,
                'representation': representative_option,
                'RMSD(OTMol)': rmsd_best,
                '#': X_A.shape[0]//3,
                'time': end_time - start_time,
                'assignment': optimal_assignment,
            }) 
        print(nameA, nameB, method, f"{rmsd_best:.2f}", f"{end_time - start_time:.2f}s")
    
    results_df = pd.DataFrame(results)
    if save == True:
        results_df.to_csv(os.path.join('./otmol_output', f'wc_{dataset_name}_{representative_option}_results.csv'), index=False)
    return results_df


def ng_experiment(mol_pair, 
               data_path: str = None,
               p_list: list = None,
               method: str = 'emd',
               reg: float = 1e-4,
               numItermax: int = 10000,
               save: bool = False, # whether to save the results
               ):
    results = []
    # Load the molecule pairs from the specified file
    for nameA, nameB in mol_pair:
        start_time = time.time()

        molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
        molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
        X_A, T_A, _ = otm.tl.process_molecule(molA) 
        X_B, T_B, _ = otm.tl.process_molecule(molB)
        optimal_assignment, rmsd_best, _ = otm.tl.cluster_alignment(
            X_A = X_A, X_B = X_B, 
            T_A = T_A, T_B = T_B,
            case = 'same element', 
            method = method, 
            p_list = p_list, 
            reg = reg, 
            numItermax = numItermax,
            save_path = f'./otmol_output/{nameB}_to_{nameA}_otmol.xyz'
        )
        
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
            #'p': p_best,
            'assignment': optimal_assignment,
            }) 
        print(nameA, nameB, method, f"{rmsd_best:.2f}", f"{end_time - start_time:.2f}s")
    
    results_df = pd.DataFrame(results)
    if save == True:
        results_df.to_csv(os.path.join('./otmol_output', f'ng_{method}_results.csv'), index=False)
    #if save == True and plain_GW:
    #    results_df.to_csv(os.path.join('./GW_output', f'ng_results.csv'), index=False)
    return results_df


def experiment(
        data_path: str = None,
        mol_pair: list = None, 
        setup: str = 'element name',
        method: str = 'fGW', 
        alpha_list: list = None,
        molecule_sizes: List[int] = None,
        dataset_name: str = None, # FGG, S1MAW1
        save: bool = False, # whether to save the results
        cst_D: float = 0.,
        ):
    results = []
    # Load the molecule pairs from the specified file
    for i, (nameA, nameB) in enumerate(mol_pair):
        if setup == 'element name':
            molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
            molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
            X_A, T_A, B_A = otm.tl.process_molecule(molA) 
            X_B, T_B, B_B = otm.tl.process_molecule(molB)
        if setup == 'atom type':
            molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
            molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
            X_A, _, B_A = otm.tl.process_molecule(molA) 
            X_B, _, B_B = otm.tl.process_molecule(molB)  
            if dataset_name == 'S1MAW1':
                _, T_A = otm.tl.parse_sy2(os.path.join(data_path, nameA + '_chimera.sy2'))
                _, T_B = otm.tl.parse_sy2(os.path.join(data_path, nameB + '_chimera.sy2'))
            else:
                _, T_A = otm.tl.parse_sy2(os.path.join(data_path, nameA + '.sy2'))
                _, T_B = otm.tl.parse_sy2(os.path.join(data_path, nameB + '.sy2'))
        if setup == 'atom connectivity':
            molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
            molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
            X_A, _, B_A = otm.tl.process_molecule(molA) 
            X_B, _, B_B = otm.tl.process_molecule(molB)
            T_A = otm.tl.parse_mna(os.path.join(data_path, nameA + '.mna'))
            T_B = otm.tl.parse_mna(os.path.join(data_path, nameB + '.mna'))
        if dataset_name == 'FGG':
            if cst_D == 0.:
                save_path = f'./otmol_output/{nameB}_to_{nameA}_otmol_{setup.split(" ")[0]}_{setup.split(" ")[1]}_c=0.xyz'
            if cst_D == 0.5:
                save_path = f'./otmol_output/{nameB}_to_{nameA}_otmol_{setup.split(" ")[0]}_{setup.split(" ")[1]}_c=0.5.xyz'
            optimal_assignment, rmsd_best, alpha_best, BCI = otm.tl.molecule_alignment(
                X_A, X_B, T_A, T_B, B_A, B_B, 
                method = method, 
                alpha_list = alpha_list, 
                molecule_sizes = molecule_sizes, 
                cst_D = cst_D,
                minimize_mismatched_edges = True,
                save_path = save_path
                )
                    
            if not otm.tl.is_permutation(T_A = T_A, T_B = T_B, perm = optimal_assignment, case = 'single'): 
                print(nameA, nameB, 'Warning: not a proper permutation')
            results.append({
                'nameA': nameA,
                'nameB': nameB,
                'RMSD(OTMol+{})'.format(setup): rmsd_best,
                '# atoms': X_A.shape[0],
                'alpha': alpha_best,
                'BCI': BCI,
                'assignment': optimal_assignment,
                }) 
            print(i, nameA, nameB, f"{rmsd_best:.2f}", f"{BCI}")
        if dataset_name == 'S1MAW1':
            if cst_D == 1:
                save_path = f'./otmol_output/{nameB}_to_{nameA}_otmol_{setup.split(" ")[0]}_{setup.split(" ")[1]}_c=1.xyz'
            optimal_assignment, rmsd_best, alpha_best, BCI = otm.tl.molecule_alignment(
                X_A, X_B, T_A, T_B, B_A, B_B, 
                method = method, 
                alpha_list = alpha_list, 
                molecule_sizes = molecule_sizes, 
                cst_D = cst_D,
                minimize_mismatched_edges = True,
                save_path = save_path
                )
            if not otm.tl.is_permutation(T_A = T_A, T_B = T_B, perm = optimal_assignment, case = 'single'): 
                print(nameA, nameB, 'Warning: not a proper permutation')
            results.append({
                'nameA': nameA,
                'nameB': nameB,
                'RMSD(OTMol+{})'.format(setup): rmsd_best,
                '# atoms': X_A.shape[0],
                'alpha': alpha_best,
                'BCI': BCI,
                'assignment': optimal_assignment,
                }) 
            print(i, nameA, nameB, f"{rmsd_best:.2f}", f"{BCI}")

    results_df = pd.DataFrame(results)
    setup = setup.replace(' ', '_')
    if save:
        results_df.to_csv(os.path.join('./otmol_output', f'{dataset_name}_{setup}_{method}_cstD={cst_D:.1f}_results.csv'), index=False)
    return pd.DataFrame(results)


# minimize mismatched edges set to False for FGG
def experiment2(
        data_path: str = None,
        mol_pair: list = None, 
        setup: str = 'element name',
        method: str = 'fGW', 
        alpha_list: list = None,
        molecule_sizes: List[int] = None,
        dataset_name: str = None, # FGG, S1MAW1
        save: bool = False, # whether to save the results
        cst_D: float = 0.,
        ):
    results = []
    # Load the molecule pairs from the specified file
    for i, (nameA, nameB) in enumerate(mol_pair):
        if setup == 'element name':
            molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
            molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
            X_A, T_A, B_A = otm.tl.process_molecule(molA) 
            X_B, T_B, B_B = otm.tl.process_molecule(molB)
        if setup == 'atom type':
            molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
            molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
            X_A, _, B_A = otm.tl.process_molecule(molA) 
            X_B, _, B_B = otm.tl.process_molecule(molB)  
            if dataset_name == 'S1MAW1':
                _, T_A = otm.tl.parse_sy2(os.path.join(data_path, nameA + '_chimera.sy2'))
                _, T_B = otm.tl.parse_sy2(os.path.join(data_path, nameB + '_chimera.sy2'))
            else:
                _, T_A = otm.tl.parse_sy2(os.path.join(data_path, nameA + '.sy2'))
                _, T_B = otm.tl.parse_sy2(os.path.join(data_path, nameB + '.sy2'))
        if setup == 'atom connectivity':
            molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
            molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
            X_A, _, B_A = otm.tl.process_molecule(molA) 
            X_B, _, B_B = otm.tl.process_molecule(molB)
            T_A = otm.tl.parse_mna(os.path.join(data_path, nameA + '.mna'))
            T_B = otm.tl.parse_mna(os.path.join(data_path, nameB + '.mna'))
        if dataset_name == 'FGG':
            if cst_D == 0.:
                save_path = f'./otmol_output/{nameB}_to_{nameA}_otmol_{setup.split(" ")[0]}_{setup.split(" ")[1]}_c=0.xyz'
            if cst_D == 0.5:
                save_path = f'./otmol_output/{nameB}_to_{nameA}_otmol_{setup.split(" ")[0]}_{setup.split(" ")[1]}_c=0.5.xyz'
            optimal_assignment, rmsd_best, alpha_best, BCI = otm.tl.molecule_alignment(
                X_A, X_B, T_A, T_B, B_A, B_B, 
                method = method, 
                alpha_list = alpha_list, 
                molecule_sizes = molecule_sizes, 
                cst_D = cst_D,
                minimize_mismatched_edges = False,
                return_BCI = True,
                #save_path = save_path
                )
                    
            if not otm.tl.is_permutation(T_A = T_A, T_B = T_B, perm = optimal_assignment, case = 'single'): 
                print(nameA, nameB, 'Warning: not a proper permutation')
            results.append({
                'nameA': nameA,
                'nameB': nameB,
                'RMSD(OTMol+{})'.format(setup): rmsd_best,
                '# atoms': X_A.shape[0],
                'alpha': alpha_best,
                'BCI': BCI,
                'assignment': optimal_assignment,
                }) 
            print(i, nameA, nameB, f"{rmsd_best:.2f}", f"{BCI}")
        if dataset_name == 'S1MAW1':
            if cst_D == 1:
                save_path = f'./otmol_output/{nameB}_to_{nameA}_otmol_{setup.split(" ")[0]}_{setup.split(" ")[1]}_c=1.xyz'
            optimal_assignment, rmsd_best, alpha_best, BCI = otm.tl.molecule_alignment(
                X_A, X_B, T_A, T_B, B_A, B_B, 
                method = method, 
                alpha_list = alpha_list, 
                molecule_sizes = molecule_sizes, 
                cst_D = cst_D,
                minimize_mismatched_edges = True,
                save_path = save_path
                )
            if not otm.tl.is_permutation(T_A = T_A, T_B = T_B, perm = optimal_assignment, case = 'single'): 
                print(nameA, nameB, 'Warning: not a proper permutation')
            results.append({
                'nameA': nameA,
                'nameB': nameB,
                'RMSD(OTMol+{})'.format(setup): rmsd_best,
                '# atoms': X_A.shape[0],
                'alpha': alpha_best,
                'BCI': BCI,
                'assignment': optimal_assignment,
                }) 
            print(i, nameA, nameB, f"{rmsd_best:.2f}", f"{BCI}")

    #results_df = pd.DataFrame(results)
    setup = setup.replace(' ', '_')
    #if save:
    #    results_df.to_csv(os.path.join('./otmol_output', f'{dataset_name}_{setup}_{method}_cstD={cst_D:.1f}_results.csv'), index=False)
    return pd.DataFrame(results)


def alpha_experiment(
        data_path: str = None,
        nameA: str = None,
        nameB: str = None,
        setup: str = 'element name',
        method: str = 'fGW', 
        alpha_list: list = np.linspace(0, 1, 101),
        dataset_name: str = None,
        cst_D: float = 0.5,
        ):  
    results = []
    if setup == 'element name':
        molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
        molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
        X_A, T_A, B_A = otm.tl.process_molecule(molA) 
        X_B, T_B, B_B = otm.tl.process_molecule(molB)
    if setup == 'atom type':
        molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
        molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
        X_A, T_A, B_A = otm.tl.process_molecule(molA) 
        X_B, T_B, B_B = otm.tl.process_molecule(molB)
        if dataset_name == 'S1':
            _, T_A = otm.tl.parse_sy2(os.path.join(data_path, nameA + '_chimera.sy2'))
            _, T_B = otm.tl.parse_sy2(os.path.join(data_path, nameB + '_chimera.sy2'))
        else:
            _, T_A = otm.tl.parse_sy2(os.path.join(data_path, nameA + '.sy2'))
            _, T_B = otm.tl.parse_sy2(os.path.join(data_path, nameB + '.sy2'))
    if setup == 'atom connectivity':
        molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
        molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
        X_A, _, B_A = otm.tl.process_molecule(molA) 
        X_B, _, B_B = otm.tl.process_molecule(molB)
        T_A = otm.tl.parse_mna(os.path.join(data_path, nameA + '.mna'))
        T_B = otm.tl.parse_mna(os.path.join(data_path, nameB + '.mna'))
    for alpha in alpha_list:
        if cst_D == 0.5:
            save_path = f'./otmol_output/{nameB}_to_{nameA}_otmol_{setup.split(" ")[0]}_{setup.split(" ")[1]}_alpha={alpha:.2f}_c=0.5.xyz'
        assignment, rmsd, _, BCI = otm.tl.molecule_alignment(
            X_A, X_B, T_A, T_B, B_A = B_A, B_B = B_B, 
            method = method, 
            alpha_list = [alpha], 
            cst_D = cst_D,
            minimize_mismatched_edges = True,
            save_path = save_path
            )
        if rmsd is None:
            print(alpha)
            continue
        results.append({
            f'RMSD(OTMol+{setup})': rmsd,
            'alpha': alpha,
            'BCI': BCI,
            'assignment': assignment,
        }) 
        #print(f"{rmsd:.2f}")
    return pd.DataFrame(results)


def p_experiment(
        data_path: str = None,
        nameA: str = None,
        nameB: str = None,
        p_list: list = None,
        ):  
    results = []
    molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
    molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
    X_A, _, _ = otm.tl.process_molecule(molA) 
    X_B, _, _ = otm.tl.process_molecule(molB)

    for p in p_list:
        assignment, rmsd, _ = otm.tl.cluster_alignment(X_A = X_A, X_B = X_B, case = 'same element', p_list = [p])
        if rmsd > 100:
            print(p)
            continue
        results.append({
            f'RMSD(OTMol)': rmsd,
            'p': p,
            'assignment': assignment,
        }) 
    return pd.DataFrame(results)


def n_trial_experiment(
        data_path: str = None,
        nameA: str = None,
        nameB: str = None,
        n_trials_list: list = None,
        ):
    results = []

    molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA)))
    molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB)))
    X_A, T_A, _ = otm.tl.process_molecule(molA) 
    X_B, T_B, _ = otm.tl.process_molecule(molB)
    for n_trials in n_trials_list:
        assignment, rmsd = otm.tl.cluster_alignment(
            X_A, X_B, T_A, T_B, case = 'molecule cluster', 
            n_atoms = 3, representative_option = 'center', n_trials = n_trials,
            save_path = f'./otmol_output/{nameB.split(".")[0]}_to_{nameA.split(".")[0]}_otmol_rep=center_n_trials={n_trials}.xyz'
            )
        results.append({
            'RMSD(OTMol)': rmsd,
            'n_trials': n_trials,
            'assignment': assignment,
        }) 
    return pd.DataFrame(results)


def cp_experiment(
        data_path: str = None,
        mol_pair: list = None, 
        method: str = 'fGW', 
        alpha_list: list = None,
        dataset_name: str = None,
        save: bool = False,
        cst_D: float = 0.,
        minimize_mismatched_edges: bool = True,
        ):
    results = []
    # Load the molecule pairs from the specified file
    for subfolder, nameA, nameB in mol_pair:
        molA = next(pybel.readfile('xyz', os.path.join(data_path, subfolder, nameA)))
        molB = next(pybel.readfile('xyz', os.path.join(data_path, subfolder, nameB)))
        X_A, T_A, B_A = otm.tl.process_molecule(molA) 
        X_B, T_B, B_B = otm.tl.process_molecule(molB)
        optimal_assignment, rmsd_best, alpha_best, BCI = otm.tl.molecule_alignment(X_A, X_B, T_A, T_B, B_A = B_A, B_B = B_B, method = method, alpha_list = alpha_list, cst_D = cst_D, minimize_mismatched_edges = True)
        if not otm.tl.is_permutation(T_A = T_A, T_B = T_B, perm = optimal_assignment, case = 'single'):
            print(nameA, nameB, 'Warning: Not a proper assignment')
        results.append({
            'nameA': nameA,
            'nameB': nameB,
            'RMSD(OTMol)': rmsd_best,
            'alpha': alpha_best,
            'BCI': BCI,
            'assignment': optimal_assignment,
        }) 
        print(nameA, nameB, f"{rmsd_best:.2f}", f"{BCI}")
    results_df = pd.DataFrame(results)
    if save:
        results_df.to_csv(os.path.join('./otmol_output', f'cp_{dataset_name}_{method}_cstD={cst_D:.1f}_results.csv'), index=False)
    return results_df


def BCI_experiment(
        data_path: str = None,
        nameA: str = None,
        nameB: str = None,
        setup: str = 'element name',
        method: str = 'fGW', 
        alpha_list: list = np.arange(0, 1, 0.01)[1:],
        dataset_name: str = None,
        cst_D: float = 0.,
        ):  
    results = []
    if setup == 'element name':
        molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
        molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
        X_A, T_A, B_A = otm.tl.process_molecule(molA) 
        X_B, T_B, B_B = otm.tl.process_molecule(molB)
    for alpha in alpha_list:
        assignment, rmsd, _, BCI = otm.tl.molecule_alignment(
            X_A, X_B, T_A, T_B, B_A = B_A, B_B = B_B, 
            method = method, 
            alpha_list = [alpha], 
            cst_D = cst_D,
            minimize_mismatched_edges = False,
            return_BCI = True,
            save_path = None
            )
        results.append({
            f'RMSD': rmsd,
            'alpha': alpha,
            'BCI': BCI*100,
            'assignment': assignment,
        }) 
        #print(f"{rmsd:.2f}")
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


def modify_pybel_coordinates(
    mol: 'pybel.Molecule',
    new_coordinates: np.ndarray,
    output
) -> 'pybel.Molecule':
    """Modify coordinates of a pybel molecule object.
    
    Parameters
    ----------
    mol : pybel.Molecule
        The pybel molecule object to modify
    new_coordinates : np.ndarray
        New coordinates array of shape (n_atoms, 3)
        
    Returns
    -------
    pybel.Molecule
        The modified molecule object
    """
    # Modify coordinates for each atom
    for i, atom in enumerate(mol.atoms):
        x, y, z = new_coordinates[i]
        atom.OBAtom.SetVector(x, y, z)
    
    mol.write("sdf", output+'.sdf', overwrite=True)


def interactive_alignment_plot_py3dmol(
    X_A: np.ndarray,
    X_B: np.ndarray,
    T_A: np.ndarray,
    T_B: np.ndarray,
    B_A: np.ndarray,
    B_B: np.ndarray,
    assignment: np.ndarray = None,
    nameA: str = 'A', 
    nameB: str = 'B', 
    #save: bool = False
    ) -> None:
    """Plot the alignment of two structures in 3D using py3Dmol.

    Parameters
    ----------
    X_A : np.ndarray
        The coordinates of the atoms in structure A.
    X_B : np.ndarray
        The coordinates of the atoms in structure B.
    T_A : np.ndarray
        The atom labels of structure A.
    T_B : np.ndarray
        The atom labels of structure B.
    B_A: np.ndarray
        The bond matrix of structure A.
    B_B: np.ndarray
        The bond matrix of structure B.
    assignment: np.ndarray
        1d array. The assignment of atoms in A to atoms in B. 
        The i-th atom in A is assigned to the assignment[i]-th atom in B.
    nameA : str
        The name of structure A.
    nameB : str
        The name of structure B.
    save : bool, optional
        Whether to save the figure as an HTML file.
    """
    import py3Dmol
    
    # Create a new viewer
    viewer = py3Dmol.view(width=800, height=600)
    
    # Convert coordinates to XYZ format for structure A
    xyz_A = f"{len(X_A)}\n{nameA}\n"
    for i, (x, y, z) in enumerate(X_A):
        atom_type = T_A[i]
        xyz_A += f"{atom_type} {x:8.3f} {y:8.3f} {z:8.3f}\n"
    
    # Convert coordinates to XYZ format for structure B
    xyz_B = f"{len(X_B)}\n{nameB}\n"
    for i, (x, y, z) in enumerate(X_B):
        atom_type = T_B[i]
        xyz_B += f"{atom_type} {x:8.3f} {y:8.3f} {z:8.3f}\n"
    
    # Add structures to viewer
    viewer.addModel(xyz_A, "xyz")
    viewer.addModel(xyz_B, "xyz")
    
    # Style for structure A 
    style_A = {"stick": {"radius": 0.06, "color": "#DC2626"},
               "sphere": {"radius": 0.15, "color": "#DC2626"}}
    
    # Style for structure B 
    style_B = {"stick": {"radius": 0.06, "color": "#2563EB"},
               "sphere": {"radius": 0.15, "color": "#2563EB"}}
    
    # Apply styles
    viewer.setStyle({"model": 0}, style_A)
    viewer.setStyle({"model": 1}, style_B)
    
    #Add bonds for structure A
    #for i in range(len(B_A)):
    #    for j in range(i+1, len(B_A)):
    #        if B_A[i,j] == 1:
    #            viewer.addCylinder({"start": {"x": X_A[i,0], "y": X_A[i,1], "z": X_A[i,2]},
    #                            "end": {"x": X_A[j,0], "y": X_A[j,1], "z": X_A[j,2]},
    #                            "color": "#DC2626", "radius": 0.06})
    
    # Add bonds for structure B
    #for i in range(len(B_B)):
    #    for j in range(i+1, len(B_B)):
    #        if B_B[i,j] == 1:
    #            viewer.addCylinder({"start": {"x": X_B[i,0], "y": X_B[i,1], "z": X_B[i,2]},
    #                            "end": {"x": X_B[j,0], "y": X_B[j,1], "z": X_B[j,2]},
    #                            "color ": "#2563EB", "radius": 0.06})
    
    # Add matching lines between atoms
    if assignment is None:
        assignment = np.arange(len(X_A), dtype=int)
        print("The assignment is not provided. Assuming identity assignment.")
    
    for i in range(len(X_A)):
        j = assignment[i]
        viewer.addCylinder({"start": {"x": X_A[i,0], "y": X_A[i,1], "z": X_A[i,2]},
                       "end": {"x": X_B[j,0], "y": X_B[j,1], "z": X_B[j,2]},
                       "color": "green", "radius": 0.03, "dashed": False})
    
    # Set camera and view
    viewer.zoomTo()
    viewer.setBackgroundColor('white')
    
    # Show the viewer
    viewer.show()


def copy_and_rename_pdb_files():
    import shutil
    # Define the base directory
    base_dir = Path("../data/extra_bio_ligands/extra_bio_ligands_boltz_af3_generated_conformers/AF3_215D_DNA_chainBC")
    
    # Get all subdirectories (excluding hidden files and the directory itself)
    subdirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(subdirs)} subdirectories")
    
    copied_count = 0
    
    for subdir in subdirs:
        # Source file path
        source_file = subdir / "model_chainA_pH74.pdb"
        
        # Destination file path (in the main AF3_215D_DNA folder)
        dest_file = base_dir / f"{subdir.name}_model_chainA_pH74.pdb"
        
        # Check if destination file already exists
        if dest_file.exists():
            print(f"Warning: {dest_file} already exists, skipping...")
            continue
        
        try:
            # Copy the file
            shutil.copy2(source_file, dest_file)
            print(f"Copied: {source_file} -> {dest_file}")
            copied_count += 1
        except Exception as e:
            print(f"Error copying {source_file}: {e}")
            skipped_count += 1
    
    print(f"\nSummary:")
    print(f"Files copied: {copied_count}")


def get_transformation_matrix(swap, reflect): # added by Xiaoqi
    """
    Creates the transformation matrix for a given swap and reflection.
    swap: tuple of indices (i,j,k) representing how to permute x,y,z
    reflect: tuple of signs (a,b,c) representing reflections along each axis
    Returns: 3x3 transformation matrix
    """
    matrix = np.zeros((3,3))
    for i in range(3):
        matrix[i, swap[i]] = reflect[i]
    return matrix


def find_assignment(X_A, X_B):
    """
    Find assignment where assignment[i] is the index of the corresponding row in X_B.
    
    Parameters:
    X_A: numpy array
    X_B: numpy array (permutation of rows of X_A)
    
    Returns:
    assignment: list where assignment[i] is the index in X_B corresponding to row i in X_A
    """
    n = X_A.shape[0]
    assignment = []
    
    for i in range(n):
        # Find which row in X_B matches row i in X_A
        for j in range(n):
            if np.array_equal(X_A[i], X_B[j]):
                assignment.append(j)
                break
    
    return np.array(assignment, dtype=int)

