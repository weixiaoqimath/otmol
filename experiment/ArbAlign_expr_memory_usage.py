# slight modification of ArbAlign.py. The algorithm is unchanged.
# the only change is the way to input information and output information.
# this script requires python 2.7
import os
import pandas as pd
from ArbAlign_expr import main, parse_molecule_pairs
import gc
import psutil


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
        print("Memory usage: %.2f MB" % memory_used)
        return result, memory_used
    return wrapper

########################################################################################


@profile_memory
def run_arbalign(xyz1_path, xyz2_path, simple=False, noHydrogens=False, verbose=False):
    # Create a simple namespace object to mimic argparse
    class Args:
        pass
    
    args = Args()
    args.xyz1 = xyz1_path
    args.xyz2 = xyz2_path
    args.simple = simple
    args.noHydrogens = noHydrogens
    args.verbose = verbose
    
    # Call the main function with our args object
    return main(args)

if __name__ == "__main__":

   if True:
      result = []
      data_path = "../data/Water-Clusters"
      mol_pair_list_path = os.path.join(data_path, 'list')
      molecule_pairs = parse_molecule_pairs(mol_pair_list_path, mol_type='water cluster')
      molecule_pairs.insert(0, molecule_pairs[0]) # add a pair for warm up
      for i, (nameA, nameB) in enumerate(molecule_pairs):
         RMSD, memory_used = run_arbalign(
         xyz1_path= os.path.join(data_path, nameA+'.xyz'),
         xyz2_path= os.path.join(data_path, nameB+'.xyz'),
         simple=False,  
         noHydrogens=False,  
         verbose=False  
         )
         if i == 0:
            continue
         result.append({
            'nameA': nameA,
            'nameB': nameB,
            'RMSD(ArbAlign)': RMSD,
            'Memory_Usage_MB': memory_used
         })
      pd.DataFrame(result).to_csv(('./arbalign_output/ArbAlignDataWC_memory_usage.csv'), index=False)

   if False:
      result = []
      data_path = "../data/Neon-Clusters"
      mol_pair_list_path = os.path.join(data_path, 'list')
      molecule_pairs = parse_molecule_pairs(mol_pair_list_path, mol_type='S1')
      molecule_pairs.insert(0, molecule_pairs[0]) # add a pair for warm up
      for i, (nameA, nameB) in enumerate(molecule_pairs):
         RMSD, memory_used = run_arbalign(
         xyz1_path= os.path.join(data_path, nameA+'.xyz'),
         xyz2_path= os.path.join(data_path, nameB+'.xyz'),
         simple=False,  
         noHydrogens=False,  
         verbose=False  
         )
         if i == 0:
            continue
         result.append({
            'nameA': nameA,
            'nameB': nameB,
            'RMSD(ArbAlign)': RMSD,
            'Memory_Usage_MB': memory_used,
         })
      pd.DataFrame(result).to_csv(('./arbalign_output/NeonCluster_memory_usage.csv'), index=False)
      