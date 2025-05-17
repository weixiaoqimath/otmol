import os
import otmol as otm
from openbabel import pybel
import pandas as pd
from experiment_utils import profile_memory

def wc_experiment(
        mol_pair, 
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
    for i, (nameA, nameB) in enumerate(mol_pair):
        rmsd_best, memory_used = run_otmol_wc(nameA, nameB, data_path, method, n_atoms, reg, numItermax, molecule_cluster_options)
        if i == 0:
            continue
        results.append({
            'nameA': nameA,
            'nameB': nameB,
            'method': method,
            'RMSD(otmol)': rmsd_best,
            'memory_used': memory_used,
            }) 
        print(nameA, nameB, method, f"{rmsd_best:.2f}", f"{memory_used:.2f}MB")
            
    results_df = pd.DataFrame(results)
    if save == True:
        results_df.to_csv(os.path.join('./otmolOutput', f'wc_memory_usage_{dataset_name}_{method}.csv'), index=False)
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
    for i, (nameA, nameB) in enumerate(mol_pair):
        rmsd_best, memory_used = run_otmol_ng(nameA, nameB, data_path, p_list, method, reg, numItermax) 
        if i == 0:
            continue      
        results.append({
            'nameA': nameA,
            'nameB': nameB,
            'method': method,
            'RMSD(otmol)': rmsd_best,
            'memory_used': memory_used,
        }) 
        print(nameA, nameB, method, f"{rmsd_best:.2f}", f"{memory_used:.2f}MB")
    results_df = pd.DataFrame(results)
    if save == True:
        results_df.to_csv(os.path.join('./otmolOutput', f'ng_memory_usage_{method}.csv'), index=False)
    return results_df


@profile_memory
def run_otmol_wc(nameA, nameB, data_path, method = 'emd', n_atoms = 3, reg = 1e-2, numItermax = 10, molecule_cluster_options = 'center'):
    molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA)))
    molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB)))
    X_A, T_A, _ = otm.tl.process_molecule(molA) 
    X_B, T_B, _ = otm.tl.process_molecule(molB)
    _, rmsd_best = otm.tl.cluster_ot_and_alignment(X_A, X_B, T_A, T_B, case = 'molecule cluster', method = method, n_atoms = n_atoms, reg = reg, numItermax = numItermax, molecule_cluster_options = molecule_cluster_options)
    return rmsd_best


@profile_memory
def run_otmol_ng(nameA, nameB, data_path, p_list, method = 'emd', reg = 1e-4, numItermax = 10000):
    molA = next(pybel.readfile('xyz', os.path.join(data_path, nameA + '.xyz')))
    molB = next(pybel.readfile('xyz', os.path.join(data_path, nameB + '.xyz')))
    X_A, T_A, _ = otm.tl.process_molecule(molA) 
    X_B, T_B, _ = otm.tl.process_molecule(molB)
    _, rmsd_best, _ = otm.tl.cluster_ot_and_alignment(X_A, X_B, T_A, T_B, case = 'same elements', method = method, p_list = p_list, reg = reg, numItermax = numItermax)
    return rmsd_best

if __name__ == "__main__":

    if False:
        data_path = "../Data/Water-Clusters"
        mol_pair_list_path = os.path.join(data_path, 'list')
        _molecule_pairs = otm.tl.parse_molecule_pairs(mol_pair_list_path, mol_type='water cluster')
        molecule_pairs = []
        for nameA, nameB in _molecule_pairs:
            molecule_pairs.append((nameA+'.xyz', nameB+'.xyz'))
        molecule_pairs.insert(0, molecule_pairs[0]) # add a pair for warm up
        if False:
            wc_experiment(molecule_pairs, data_path, dataset_name = 'ArbAlignDataWC')
        if True:
            wc_experiment(molecule_pairs, data_path, method = 'sOT', dataset_name = 'ArbAlignDataWC')

    if False:
        group1_data_path = "../Data/Our_Benchmark_20250410_ver1/Water_Cluster_3_30/water_xyz_output_1st_2nd"
        group1_info = pd.read_csv('../Data/Our_Benchmark_20250410_ver1/Water_Cluster_3_30/water_cluster_1st_2nd_lowest_energy.csv')
        molecule_pairs = list(zip(group1_info['Reference'], group1_info['Target']))
        molecule_pairs.insert(0, molecule_pairs[0]) # add a pair for warm up
        if False:
            wc_experiment(molecule_pairs, group1_data_path, dataset_name = '1st2nd')
        if True:
            wc_experiment(molecule_pairs, group1_data_path, method = 'sOT', dataset_name = '1st2nd')

    if False:
        group2_data_path = "../Data/Our_Benchmark_20250410_ver1/Water_Cluster_3_30/water_xyz_output_1st_to_20th"
        group2_info = pd.read_csv('../Data/Our_Benchmark_20250410_ver1/Water_Cluster_3_30/water_cluster_largest_RMSD_pair_among_20_lowest_energy.csv')
        molecule_pairs = list(zip(group2_info['Reference'], group2_info['Target']))
        molecule_pairs.insert(0, molecule_pairs[0]) # add a pair for warm up
        if False:
            wc_experiment(molecule_pairs, group2_data_path, dataset_name = 'Largest_RMSD')
        if True:
            wc_experiment(molecule_pairs, group2_data_path, method = 'sOT', dataset_name = 'Largest_RMSD')

    if True:
        data_path = "../Data/Neon-Clusters"
        mol_pair_list_path = os.path.join(data_path, 'list')
        molecule_pairs = otm.tl.parse_molecule_pairs(mol_pair_list_path, mol_type='S1')
        molecule_pairs.insert(0, molecule_pairs[0]) # add a pair for warm up
        if False:
            ng_experiment(molecule_pairs, data_path, p_list=range(2,9), method = 'emd')
        if True:
            ng_experiment(molecule_pairs, data_path, p_list=range(2,9), method = 'sinkhorn')