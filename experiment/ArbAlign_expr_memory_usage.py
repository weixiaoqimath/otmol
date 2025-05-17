# slight modification of ArbAlign.py. The algorithm is unchanged.
# the only change is the way to input information and output information.
# this script requires python 2.7
import os
import sys
import numpy as np
import hungarian
from collections import Counter 
import operator
import argparse
import pandas as pd
from ArbAlign_expr import main, parse_molecule_pairs
import resource
import platform
import gc
import psutil


#def get_memory_usage():
#    """Get memory usage in MB, handling platform differences"""
#    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss # the maximum resident set size utilized (in bytes).
    # On macOS, ru_maxrss is in bytes, on Linux it's in KB
#    if platform.system() == 'Darwin':  # macOS
#        return mem / 1024.0**2  # Convert bytes to MB
#    return mem / 1024.0  # Already in MB on Linux


#def profile_memory(func):
#    """Simple memory profiler decorator for Python 2.7"""
#    def wrapper(*args, **kwargs):
#        # Force garbage collection before measurement
#        gc.collect()
#        start_mem = get_memory_usage()
#        result = func(*args, **kwargs)
#        end_mem = get_memory_usage()
#        memory_used = end_mem - start_mem
#        print("Memory usage: %.2f MB" % memory_used)
#        return result, memory_used
#    return wrapper


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

   if False:
      result = []
      data_path = "../Data/Water-Clusters"
      mol_pair_list_path = os.path.join(data_path, 'list')
      molecule_pairs = parse_molecule_pairs(mol_pair_list_path, mol_type='water cluster')
      molecule_pairs.insert(0, molecule_pairs[0]) # add a pair for warm up
      for i, (nameA, nameB) in enumerate(molecule_pairs):
         RMSD, memory_used = run_arbalign(
         xyz1_path= os.path.join(data_path, nameA+'.xyz'),
         xyz2_path= os.path.join(data_path, nameB+'.xyz'),
         simple=False,  # Set to True for faster but less thorough alignment
         noHydrogens=False,  # Set to True to ignore hydrogen atoms
         verbose=False  # Set to True to see detailed output
         )
         if i == 0:
            continue
         result.append({
            'nameA': nameA,
            'nameB': nameB,
            'RMSD(ArbAlign)': RMSD,
            'Memory_Usage_MB': memory_used
         })
      pd.DataFrame(result).to_csv(('./ArbAlignOutput/ArbAlignDataWC_memory_usage.csv'), index=False)

   if False:
      result = []
      group1_data_path = "../Data/Our_Benchmark_20250410_ver1/Water_Cluster_3_30/water_xyz_output_1st_2nd"
      group1_info = pd.read_csv('../Data/Our_Benchmark_20250410_ver1/Water_Cluster_3_30/water_cluster_1st_2nd_lowest_energy.csv')
      list_of_pairs = list(zip(group1_info['Reference'], group1_info['Target']))
      list_of_pairs.insert(0, list_of_pairs[0]) # add a pair for warm up
      for i, (nameA, nameB) in enumerate(list_of_pairs):
         RMSD, memory_used = run_arbalign(
         xyz1_path= os.path.join(group1_data_path, nameA),
         xyz2_path= os.path.join(group1_data_path, nameB),
         simple=False,  # Set to True for faster but less thorough alignment
         noHydrogens=False,  # Set to True to ignore hydrogen atoms
         verbose=False  # Set to True to see detailed output
            )
         if i == 0:
            continue
         result.append({
                'nameA': nameA,
                'nameB': nameB,
                'RMSD(ArbAlign)': RMSD,
                'Memory_Usage_MB': memory_used,
         })
      pd.DataFrame(result).to_csv(('./ArbAlignOutput/1st2ndWC_memory_usage.csv'), index=False)
   
   if False:
      result = []
      group2_data_path = "../Data/Our_Benchmark_20250410_ver1/Water_Cluster_3_30/water_xyz_output_1st_to_20th"
      group2_info = pd.read_csv('../Data/Our_Benchmark_20250410_ver1/Water_Cluster_3_30/water_cluster_largest_RMSD_pair_among_20_lowest_energy.csv')
      list_of_pairs = list(zip(group2_info['Reference'], group2_info['Target']))
      list_of_pairs.insert(0, list_of_pairs[0]) # add a pair for warm up
      for i, (nameA, nameB) in enumerate(list_of_pairs):
         RMSD, memory_used = run_arbalign(
         xyz1_path= os.path.join(group2_data_path, nameA),
         xyz2_path= os.path.join(group2_data_path, nameB),
         simple=False,  # Set to True for faster but less thorough alignment
         noHydrogens=False,  # Set to True to ignore hydrogen atoms
         verbose=False  # Set to True to see detailed output
         )
         if i == 0:
            continue
         result.append({
            'nameA': nameA,
            'nameB': nameB,
            'RMSD(ArbAlign)': RMSD,
            'Memory_Usage_MB': memory_used,
         })
      pd.DataFrame(result).to_csv(('./ArbAlignOutput/largest_RMSD_WC_memory_usage.csv'), index=False)

   if True:
      result = []
      data_path = "../Data/Neon-Clusters"
      mol_pair_list_path = os.path.join(data_path, 'list')
      molecule_pairs = parse_molecule_pairs(mol_pair_list_path, mol_type='S1')
      molecule_pairs.insert(0, molecule_pairs[0]) # add a pair for warm up
      for i, (nameA, nameB) in enumerate(molecule_pairs):
         RMSD, memory_used = run_arbalign(
         xyz1_path= os.path.join(data_path, nameA+'.xyz'),
         xyz2_path= os.path.join(data_path, nameB+'.xyz'),
         simple=False,  # Set to True for faster but less thorough alignment
         noHydrogens=False,  # Set to True to ignore hydrogen atoms
         verbose=False  # Set to True to see detailed output
         )
         if i == 0:
            continue
         result.append({
            'nameA': nameA,
            'nameB': nameB,
            'RMSD(ArbAlign)': RMSD,
            'Memory_Usage_MB': memory_used,
         })
      pd.DataFrame(result).to_csv(('./ArbAlignOutput/NeonCluster_memory_usage.csv'), index=False)
      