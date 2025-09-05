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

## Documentation

See detailed documentation and examples at [https://otmol.readthedocs.io/en/latest/index.html](https://otmol.readthedocs.io/en/latest/index.html).

## Reference

https://arxiv.org/abs/2509.01550