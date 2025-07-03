# slight modification of ArbAlign.py. The algorithm is unchanged.
# the only change is the way to input information and output information.
# this scripts requires python 2.7
import os
import pandas as pd
from ArbAlign_expr import run_arbalign

if __name__ == "__main__":
   if True:
      data_path = "../Data/first_8_molecules_RDKIT"
      lig_list = [
         '1ln1_DLP', '4csv_imatinib', '5bvs_EIC', '6ln3_ATP', '6y13_stapledHelix', '8w4x_BGC', '8w4x_BGCGLC', '215d_DNA']
      for ligand in lig_list:
         result = []
         for i in range(50):
            res, _, _ = run_arbalign(
                xyz1_path= os.path.join(data_path, 'extra_bio_ligands', ligand+'.xyz'),
                xyz2_path= os.path.join(data_path, 'extra_bio_ligands_RDKIT', ligand, ligand+'_conf'+str(i)+'.xyz'),
                simple=False,  # Set to True for faster but less thorough alignment
                noHydrogens=True,  # Set to True to ignore hydrogen atoms
                verbose=False  # Set to True to see detailed output
            )
            result.append({
                'nameA': ligand,
                'nameB': ligand+'_conf'+str(i),
                'RMSD(ArbAlign)': res,
            })            
         pd.DataFrame(result).to_csv(('./arbalign_output/{}_result.csv'.format(ligand)), index=False)


      