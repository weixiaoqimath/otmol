import otmol as otm
from openbabel import pybel   
import time
import pandas as pd
import os
from typing import List
import gc
import psutil
import numpy as np
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
        #if plain_GW:
        #    rmsd = GW_alignment(X_A, X_B, T_A, T_B)
        #    results.append({
        #        'nameA': nameA,
        #        'nameB': nameB,
        #        'RMSD(GW)': rmsd,
        #        '# atoms': X_A.shape[0],
        #    }) 
        #    print(nameA, nameB, f"{rmsd:.2f}")
        #else:
        optimal_assignment, rmsd_best = otm.tl.cluster_alignment(
            X_A, X_B, T_A, T_B, case = 'molecule cluster', 
            method = method, n_atoms = n_atoms, 
            reg = reg, numItermax = numItermax, 
            representative_option = representative_option,
            n_trials = n_trials)
            
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
               plain_GW: bool = False,
               n_trials: int = 100,
               perturbation: bool = False,
               ):
    results = []
    # Load the molecule pairs from the specified file
    for nameA, nameB in mol_pair:
        start_time = time.time()

        molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
        molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
        X_A, T_A, _ = otm.tl.process_molecule(molA) 
        X_B, T_B, _ = otm.tl.process_molecule(molB)
        if not perturbation:
            optimal_assignment, rmsd_best, p_best = otm.tl.cluster_alignment(X_A = X_A, X_B = X_B, case = 'same element', method = method, p_list = p_list, reg = reg, numItermax = numItermax)
        else:
            optimal_assignment, rmsd_best = otm.tl.perturbation_before_gw(X_A = X_A, X_B = X_B, n_trials = n_trials, return_best = True, scale = None)
        
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
        #if plain_GW:
        #    rmsd = GW_alignment(X_A, X_B, T_A, T_B)
        #    results.append({
        #        'nameA': nameA,
        #        'nameB': nameB,
        #        'RMSD(GW+{})'.format(setup): rmsd,
        #        '# atoms': X_A.shape[0],
        #    }) 
        #    print(nameA, nameB, f"{rmsd:.2f}")
        #else:
        optimal_assignment, rmsd_best, alpha_best = otm.tl.molecule_alignment(
            X_A, X_B, T_A, T_B, B_A, B_B, 
            method = method, 
            alpha_list = alpha_list, 
            molecule_sizes = molecule_sizes, 
            cst_D = cst_D
            )
                
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
        print(i, nameA, nameB, f"{rmsd_best:.2f}")

    results_df = pd.DataFrame(results)
    setup = setup.replace(' ', '_')
    if save:
        results_df.to_csv(os.path.join('./otmol_output', f'{dataset_name}_{setup}_{method}_cstD={cst_D:.1f}_results.csv'), index=False)
        #else:
        #    results_df.to_csv(os.path.join('./otmol_output', f'{dataset_name}_{setup}_{method[0]}_{method[1]}_results.csv'), index=False)
    #if save and plain_GW:
    #    if len(method) == 1:
    #        results_df.to_csv(os.path.join('./GW_output', f'{dataset_name}_{setup}_results.csv'), index=False)
    #    else:
    #        results_df.to_csv(os.path.join('./GW_output', f'{dataset_name}_{setup}_{method[0]}_{method[1]}_results.csv'), index=False)
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
        assignment, rmsd, _ = otm.tl.molecule_alignment(X_A, X_B, T_A, T_B, B_A = B_A, B_B = B_B, method = method, alpha_list = [alpha], cst_D = cst_D)
        if rmsd > 100:
            print(alpha)
            continue
        results.append({
            f'RMSD(OTMol+{setup})': rmsd,
            'alpha': alpha,
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
            n_atoms = 3, representative_option = 'center', n_trials = n_trials)
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
        ):
    results = []
    # Load the molecule pairs from the specified file
    for subfolder, nameA, nameB in mol_pair:
        molA = next(pybel.readfile('xyz', os.path.join(data_path, subfolder, nameA)))
        molB = next(pybel.readfile('xyz', os.path.join(data_path, subfolder, nameB)))
        X_A, T_A, B_A = otm.tl.process_molecule(molA) 
        X_B, T_B, B_B = otm.tl.process_molecule(molB)
        #if plain_GW:
        #    rmsd = GW_alignment(X_A, X_B, T_A, T_B)
        #    results.append({
        #        'nameA': nameA,
        #        'nameB': nameB,
        #        'RMSD(GW)': rmsd,
        #        '# atoms': X_A.shape[0],
        #    }) 
        #    print(nameA, nameB, f"{rmsd:.2f}")
        #else:
        optimal_assignment, rmsd_best, alpha_best = otm.tl.molecule_alignment(X_A, X_B, T_A, T_B, B_A = B_A, B_B = B_B, method = method, alpha_list = alpha_list, cst_D = cst_D)
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
    results_df = pd.DataFrame(results)
    if save:
        results_df.to_csv(os.path.join('./otmol_output', f'cp_{dataset_name}_{method}_cstD={cst_D:.1f}_results.csv'), index=False)
    #if save and plain_GW:
    #    results_df.to_csv(os.path.join('./GW_output', f'cp_{dataset_name}_results.csv'), index=False)
    return results_df


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
    save: bool = False
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
    if save:
        viewer.write_html(f"{nameA}_{nameB}.html")


def GW_alignment(
    X_A: np.ndarray,
    X_B: np.ndarray,
    T_A: np.ndarray,
    T_B: np.ndarray,
    p_list: list = range(2, 9),
    ) -> None:
    """
    Align the two structures only on the same element using GW.
    """
    import ot
    from scipy.spatial import distance_matrix
    label_list = np.unique(T_A)
    X_A_label_list = []
    permuted_X_B_label_list = []
    for label in label_list:
        X_A_label = X_A[T_A == label]
        X_B_label = X_B[T_B == label]
        X_A_label_list.append(X_A_label)
        perm_best = None
        rmsd_best = 1e10
        if len(X_A_label) == 1 and len(X_B_label) == 1:
            permuted_X_B_label_list.append(X_B_label)
            continue
        if len(X_A_label) == 2 and len(X_B_label) == 2:
            for perm in [np.array([0, 1]), np.array([1, 0])]:
                X_B_label_aligned, _, _ = otm.tl.kabsch(X_A_label, X_B_label, otm.tl.permutation_to_matrix(perm))
                rmsd = otm.tl.root_mean_square_deviation(X_A_label, X_B_label_aligned)
                if rmsd < rmsd_best:
                    rmsd_best = rmsd
                    perm_best = perm
            permuted_X_B_label_list.append(X_B_label[perm_best])
            continue
        for p in p_list:
            D_A = distance_matrix(X_A_label, X_A_label)**p
            D_B = distance_matrix(X_B_label, X_B_label)**p
            P = ot.gromov.gromov_wasserstein(D_A/D_A.max(), D_B/D_A.max(), symmetric=True)
            perm = np.argmax(P, axis=1)
            if len(np.unique(perm)) != len(X_A_label):
                continue
            X_B_label_aligned, _, _ = otm.tl.kabsch(X_A_label, X_B_label, otm.tl.permutation_to_matrix(perm))
            rmsd = otm.tl.root_mean_square_deviation(X_A_label, X_B_label_aligned)
            if rmsd < rmsd_best:
                rmsd_best = rmsd
                perm_best = perm
        permuted_X_B_label_list.append(X_B_label[perm_best])

    _X_A, _X_B = np.vstack(X_A_label_list), np.vstack(permuted_X_B_label_list)
    _X_B_aligned, _, _ = otm.tl.kabsch(_X_A, _X_B, np.eye(len(_X_A)))
    rmsd = otm.tl.root_mean_square_deviation(_X_A, _X_B_aligned)
    return rmsd

