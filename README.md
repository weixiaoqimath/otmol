# OTMol

Molecular alignment with optimal transport

## Installation

OTMol depends on the following packages:

- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0
- pot >= 0.9.5
- networkx >= 3.1
- plotly >= 5.15.0

OTMol has been tested in python 3.9 virtual environment.

```
git clone https://github.com/weixiaoqimath/otmol.git
cd otmol
pip install .
```

## Usage

```
import numpy as np
import otmol as otm
from openbabel import pybel
import os
data_path = "../Data/FGG-Tripeptide/"
nameA = '252_FGG55.xyz'
nameB = '258_FGG224.xyz'
molA = next(pybel.readfile("xyz", os.path.join(data_path, nameA)))
molB = next(pybel.readfile("xyz", os.path.join(data_path, nameB)))
X_A, T_A, B_A = otm.tl.process_molecule(molA) 
X_B, T_B, B_B = otm.tl.process_molecule(molB)
```

## Ackonwledgement

If you find this work useful, please cite:
