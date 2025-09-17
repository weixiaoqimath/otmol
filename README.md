# OTMol

Molecular alignment with optimal transport

## Installation


```
git clone https://github.com/weixiaoqimath/otmol.git
cd otmol
conda env create -f environment.yml
pip install .
```
OTMol has been tested in python 3.9 virtual environment.

## Usage

For two single molecules, OTMol requires the user to provide three inputs for each molecule: the coordinate array X (an (n, 3) array), the atom label array T (an (n, ) array), and the adjacency matrix B (an (n, n) array) that represents the graph structure. If the molecule is loaded by openbabel, the user can get them by otmol.tl.process_molecule().
```
import numpy as np
import otmol as otm
from openbabel import pybel
import os
data_path = "./data/FGG-Tripeptide/"
nameA = '252_FGG55.xyz'
nameB = '258_FGG224.xyz'
molA = next(pybel.readfile("xyz", os.path.join(data_path, nameA)))
molB = next(pybel.readfile("xyz", os.path.join(data_path, nameB)))
X_A, T_A, B_A = otm.tl.process_molecule(molA) 
X_B, T_B, B_B = otm.tl.process_molecule(molB)
alpha_list = np.arange(0, 1.0, 0.01)[1:]
assignment, rmsd_best, alpha_best, BCI = otm.tl.molecule_alignment(
    X_A, X_B, T_A, T_B, B_A, B_B, cst_D = 0.5,
    alpha_list = alpha_list, minimize_mismatched_edges = True)
print('The RMSD of the alignment is', rmsd_best)
print('The atom assignment is', assignment)
print('The BCI of the alignment is {}%'.format(BCI*100))
```
The assignment is returned as an integer list 'assignment', where j = assignment[i] means the i-th atom in A is assigned to the j-th atom in B. The user can save the aligned coordinates of B by specifying a output path with the argument "save_path" in molecule_alignment(). OTMol will output an xyz file where atoms in B are permuted according to the assignment.


## Documentation

See detailed documentation and examples at [https://otmol.readthedocs.io/en/latest/index.html](https://otmol.readthedocs.io/en/latest/index.html).

## Reference

https://arxiv.org/abs/2509.01550