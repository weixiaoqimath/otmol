import numpy as np
from openbabel import openbabel
from scipy.spatial import distance_matrix
from scipy.sparse import csr_array
from scipy.sparse.csgraph import floyd_warshall

ATOMIC_NAME = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br"
}

ATOMIC_COLOR = {
    "H": "silver",
    "C": "black",
    "N": "blue",
    "O": "red",
    "F": "green",
    "P": "orange",
    "S": "yellow",
    "Cl": "limegreen",
    "Br": "salmon"
}

ATOMIC_SIZE = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20
}

ATOMIC_PROPERTIES = {
    "H": {"en": 2.20, "vdw": 1.10, "cov": 0.32},   # Hydrogen (H)
    "C": {"en": 2.55, "vdw": 1.70, "cov": 0.76},   # Carbon (C)
    "N": {"en": 3.04, "vdw": 1.55, "cov": 0.71},   # Nitrogen (N)
    "O": {"en": 3.44, "vdw": 1.52, "cov": 0.66},   # Oxygen (O)
    "F": {"en": 3.98, "vdw": 1.47, "cov": 0.64},   # Fluorine (F)
    "P": {"en": 2.19, "vdw": 1.80, "cov": 1.06},  # Phosphorus (P)
    "S": {"en": 2.58, "vdw": 1.80, "cov": 1.02},  # Sulfur (S)
    "Cl": {"en": 3.16, "vdw": 1.75, "cov": 0.99},  # Chlorine (Cl)
    "Br": {"en": 2.96, "vdw": 1.85, "cov": 1.14},  # Bromine (Br)
}


def geodesic_distance(
    X,
    B
):
    """
    B: adjacency matrix of the graph
    """
    dists = distance_matrix(X, X)   
    graph = np.where(B, dists, 0)
    graph = csr_array(graph)
    geodesic = floyd_warshall(graph, directed=False)
    return geodesic


def atom_physchem_distance(atomic_type1: str, atomic_type2: str) -> float:
    """
    Computes normalized Euclidean distance between two atomic numbers using:
    - Electronegativity (en)
    - Van der Waals radius (vdw)
    - Covalent radius (cov)
    """
    if atomic_type1 not in ATOMIC_PROPERTIES or atomic_type2 not in ATOMIC_PROPERTIES:
        raise ValueError(f"Unsupported atomic type(s). Supported: {list(ATOMIC_PROPERTIES.keys())}")

    p1 = ATOMIC_PROPERTIES[atomic_type1]
    p2 = ATOMIC_PROPERTIES[atomic_type2]

    # Get normalization ranges from the 9 elements
    en_values = [v["en"] for v in ATOMIC_PROPERTIES.values()]
    vdw_values = [v["vdw"] for v in ATOMIC_PROPERTIES.values()]
    cov_values = [v["cov"] for v in ATOMIC_PROPERTIES.values()]

    def normalize(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5

    en_min, en_max = min(en_values), max(en_values)     # 2.20 (H) to 3.98 (F)
    vdw_min, vdw_max = min(vdw_values), max(vdw_values) # 1.10 (H) to 1.85 (Br)
    cov_min, cov_max = min(cov_values), max(cov_values) # 0.32 (H) to 1.14 (Br)

    # Normalize properties
    en1 = normalize(p1["en"], en_min, en_max)
    vdw1 = normalize(p1["vdw"], vdw_min, vdw_max)
    cov1 = normalize(p1["cov"], cov_min, cov_max)

    en2 = normalize(p2["en"], en_min, en_max)
    vdw2 = normalize(p2["vdw"], vdw_min, vdw_max)
    cov2 = normalize(p2["cov"], cov_min, cov_max)

    # Euclidean distance
    return np.sqrt((en1 - en2)**2 + (vdw1 - vdw2)**2 + (cov1 - cov2)**2)

def molecule_physchem_distance(
    T1,
    T2
):
    n1 = T1.shape[0]
    n2 = T2.shape[0]
    C = np.empty([n1,n2], dtype = float)
    for i in range(n1):
        for j in range(n2):
            C[i,j] = atom_physchem_distance(T1[i], T2[j])
    return C
