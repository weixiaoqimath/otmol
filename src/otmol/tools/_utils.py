import numpy as np
from scipy.spatial import distance_matrix
from typing import List, Tuple, Union, Optional


def root_mean_square_deviation(
        X: np.ndarray, 
        Y: np.ndarray
        ) -> float:
    """Compute the Root Mean Square Deviation (RMSD) between two sets of points.

    Parameters
    ----------
    X : np.ndarray
        First set of points as an ``n`` x ``3`` array.
    Y : np.ndarray
        Second set of points as an ``n`` x ``3`` array.

    Returns
    -------
    float
        The RMSD between the two sets of points.
    """
    return np.sqrt(np.mean(np.sum((X - Y) ** 2, axis=1)))


def cost_matrix(
        X_A: np.ndarray = None, 
        X_B: np.ndarray = None, 
        T_A: np.ndarray = None, 
        T_B: np.ndarray = None, 
        k: float = 1e12, 
        ) -> np.ndarray:
    """Create a cost matrix for T_A and T_B.

    T_A and T_B are the atom labels.
    If X_A and X_B are provided, creates a cost matrix where the cost of atoms having the same label
    is the Euclidean distance between the atoms and the cost of atoms having different labels is a constant k 
    (can be set to infinity).
    If X_A and X_B are not provided, creates a cost matrix where the cost of atoms having the same label
    is 0 and the cost of atoms having different labels is a constant k (can be set to infinity).
    If multiple_molecules_block_size is provided, creates a block diagonal matrix where off-diagonal blocks
    are filled with k.

    Parameters
    ----------
    X_A : numpy.ndarray
        Array of coordinates for molecule A.
    X_B : numpy.ndarray
        Array of coordinates for molecule B.
    T_A : numpy.ndarray
        Array of atom labels for molecule A.
    T_B : numpy.ndarray
        Array of atom labels for molecule B.
    k : float, optional
        Cost of mismatching atoms, by default 1e11.

    Returns
    -------
    numpy.ndarray
        Cost matrix as numpy array.
    """
    n = len(T_A)
    m = len(T_B)
    C = np.full((n, m), k)
    
    for i in range(n):
        for j in range(m):
            if T_A[i] == T_B[j]:
                if X_A is not None and X_B is not None:
                    C[i, j] = np.linalg.norm(X_A[i] - X_B[j])
                else:
                    C[i, j] = 0
    return C


def compare_labels(
        list1, 
        list2
        ) -> List[Tuple[int, str, str]]:
    """Compare two arrays of atom labels and return the indices and labels where they differ.

    Parameters
    ----------
    list1 : array_like
        Array of atom labels.
    list2 : array_like
        Array of atom labels.

    Returns
    -------
    List[Tuple[int, str, str]]
        A list of tuples, where each tuple contains the index and the differing labels
        in the format (index, label_from_list1, label_from_list2).
    """
    differences = []
    # Compare elements up to the length of the shorter list
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            differences.append((i, list1[i], list2[i]))

    return differences


def parse_molecule_pairs(
        file_path: str, 
        mol_type: str = 'water cluster'
        ) -> List[List[str]]:
    """Parses list file in ArbAlign data folder.

    Parameters:
    ----------
    file_path : str
        Path to the file containing molecule pairs.

    Returns:
    -------
    list of lists
        A list where each element is a pair [molA, molB].
    """
    molecule_pairs = []
    with open(file_path, 'r') as file:
        if mol_type == 'water cluster' or mol_type == 'S1':
            for line in file:
                line = line.strip()  # Remove any leading/trailing whitespace
                if line:  # Skip empty lines
                    # Some lines are like "molA_molB_2", and some are like "molA_molB"
                    molecule_pairs.append([line.split('_')[0], line.split('_')[1]])
        if mol_type == 'FGG':
            next(file)  # Skip the first line
            for line in file:
                line = line.strip()
                if line:
                    molA, molB = line.split('-')
                    molecule_pairs.append([molA, molB])
    return molecule_pairs


def permutation_to_matrix(permutation, n: int = None, m: int = None) -> np.ndarray:
    """Convert a permutation list to a permutation matrix.

    Parameters
    ----------
    permutation : array_like
        A list or array representing the permutation.
        For example, [2, 0, 1] means index 0 maps to 2, index 1 maps to 0, etc.
    n : int, optional
        The number of rows in the matrix.
    m : int, optional
        The number of columns in the matrix.

    Returns
    -------
    numpy.ndarray
        A permutation matrix where matrix[i, j] = 1 if permutation[i] = j, 0 otherwise.
    """
    if n is None and m is None:
        n = len(permutation)
        matrix = np.zeros((n, n), dtype=int)
    else:
        matrix = np.zeros((n, m), dtype=int)

    for i, j in enumerate(permutation):
        matrix[i, j] = 1
    return matrix


def is_permutation(
        T_A: np.ndarray = None,
        T_B: np.ndarray = None,
        perm: np.ndarray = None, 
        case: str = None, 
        n_atoms: int = None
        ) -> bool:
    """Check if the given array is a permutation, with optional special cases.

    Parameters
    ----------
    T_A : numpy.ndarray
        Array like. A 1D array of atom labels.
    T_B : numpy.ndarray
        Array like. A 1D array of atom labels.
    perm : numpy.ndarray
        Array like. A 1D array of integers.
    case : str, optional
        Special case to check:
            - None: Basic permutation check (all elements unique).
            - 'single': labels are matched.
            - 'molecule cluster': molecule cluster permutation (groups of n_atoms).
    n_atoms : int, optional
        Only used when case is 'molecule cluster'. The number of atoms in a molecule.

    Returns
    -------
    bool
        True if the array satisfies the permutation condition, False otherwise.
    """
    if case is None:
        return len(np.unique(perm)) == len(perm)
    if case == 'single':
        return len(np.unique(perm)) == len(perm) and np.array_equal(T_A, T_B[perm])
    if case == 'molecule cluster':
        return is_molecule_cluster_permutation(T_A = T_A, T_B = T_B, perm = perm, n_atoms = n_atoms)


def is_molecule_cluster_permutation(
        T_A: np.ndarray = None, 
        T_B: np.ndarray = None, 
        perm: np.ndarray = None, 
        n_atoms: int = 3
        ) -> bool:
    """Check if the given array perm is a permutation that satisfies the condition:
    After grouping every n_atoms = k integers, sorting each group in ascending order
    results in [min(group), min(group)+1, ..., min(group)+k-1], and min(group) is a multiple of k.

    Parameters
    ----------
    T_A : array like
        A 1D numpy array of atom labels.
    T_B : array like
        A 1D numpy array of atom labels.
    perm : array like
        A 1D numpy array of integers.
    n_atoms : int
        The number of atoms in a molecule.

    Returns
    -------
    bool
        True if all groups satisfy the condition, False otherwise.
    """
    # check label consistency
    if T_A is not None and T_B is not None:
        if not np.array_equal(T_A, T_B[perm]):
            return False
    # basic check
    if len(np.unique(perm)) != len(perm):
        return False
    n = len(perm)
    if n % n_atoms != 0:
        raise ValueError("The length of the array must be a multiple of {}.".format(n_atoms))
    
    blocks = [perm[i:i+n_atoms] for i in range(0, n, n_atoms)]  # Group every n_atoms integers
    for block in blocks:
        sorted_block = sorted(block)
        if sorted_block[0] % n_atoms != 0:
            return False
        if not np.array_equal(sorted_block, np.arange(sorted_block[0], sorted_block[0] + n_atoms)):
            return False
    return True


def add_perturbation(
        X: np.ndarray, 
        noise_scale: float = None, 
        random_state: int = None
        ) -> np.ndarray:
    """Add Gaussian noise to coordinates.
    
    Parameters
    ----------
    X : numpy.ndarray
        The 3D coordinates as an n x 3 array.
    noise_scale : float, optional
        The standard deviation of the Gaussian noise to add.
        If None, it will be set to 1/10 of the shortest distance between coordinates.
    random_state : int or numpy.random.RandomState
        Seed for the random number generator or RandomState instance.
        
    Returns
    -------
    numpy.ndarray
        The perturbed coordinates.
    """
    if noise_scale is None:
        # Calculate pairwise distances between all points
        distances = distance_matrix(X, X)
        # Set diagonal to infinity to ignore self-distances
        np.fill_diagonal(distances, np.inf)
        # Find minimum distance and set noise scale to 1/10 of that
        min_distance = np.min(distances)
        noise_scale = min_distance / 10.0
    
    # Create random state
    rng = np.random.RandomState(random_state)
    noise = rng.normal(0, noise_scale, X.shape)
    return X + noise

def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize a matrix by dividing by its maximal non-infinity value.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        Input matrix that may contain infinity values.
        
    Returns
    -------
    numpy.ndarray
        Normalized matrix where non-infinity values are divided by the maximum non-infinity value.
        Infinity values remain unchanged.
    """
    # Create a mask for non-infinity values
    finite_mask = np.isfinite(matrix)
    
    if not np.any(finite_mask):
        return matrix  # Return original matrix if all values are infinity
        
    # Find maximum non-infinity value
    max_val = np.max(matrix[finite_mask])
    
    if max_val == 0:
        return matrix  # Return original matrix if max value is 0
        
    # Create normalized matrix
    normalized = matrix.copy()
    normalized[finite_mask] = normalized[finite_mask] / max_val
    
    return normalized


def resolve_sinkhorn_conflicts(P: np.ndarray) -> List[np.ndarray]:
    """Resolve conflicts in a Sinkhorn transport plan P.

    Parameters
    ----------
    P : numpy.ndarray
        The Sinkhorn transport plan to resolve conflicts in.

    Returns
    -------
    list of numpy.ndarray
        A list of resolved Sinkhorn transport plans.
    """
    n = P.shape[0]
    assignment = np.full(n, -1)
    used = np.zeros(n, dtype=bool)
    
    # First pass: assign unambiguous cases
    for i in range(n):
        row = P[i]
        max_val = np.max(row)
        max_indices = np.where(row == max_val)[0]
        if len(max_indices) == 1 and not used[max_indices[0]]:
            assignment[i] = max_indices[0]
            used[max_indices[0]] = True
    
    if np.sum(assignment == -1) > 2:
        return None
    else:
        res = []
        unassigned = np.where(assignment == -1)[0]
        unused = np.where(~used)[0]

        tmp = assignment.copy()
        tmp[unassigned] = unused
        res.append(tmp)

        tmp = assignment.copy()
        tmp[unassigned] = unused[::-1]
        res.append(tmp)

        return res


def add_molecule_indices(
        T_A: List[str],
        T_B: List[str],
        molecule_sizes: List[int]
        ) -> Tuple[List[str], List[str]]:
    """Add indices to labels based on molecule sizes.
    
    For each molecule, adds an index to all labels in that molecule's range.
    For example, if molecule_sizes = [3, 2], then:
    - Labels 0-2 get index 0
    - Labels 3-4 get index 1
    
    Parameters
    ----------
    T_A : List[str]
        List of labels for molecule A
    T_B : List[str]
        List of labels for molecule B
    molecule_sizes : List[int]
        List of sizes for each molecule. Sum must equal len(T_A) = len(T_B)
        
    Returns
    -------
    T_A_with_indices : np.ndarray
        Array of labels for molecule A with indices
    T_B_with_indices : np.ndarray
        Array of labels for molecule B with indices
        
    Raises
    ------
    ValueError
        If sum(molecule_sizes) != len(T_A) or len(T_A) != len(T_B)
    """
    if sum(molecule_sizes) != len(T_A) or len(T_A) != len(T_B):
        raise ValueError("sum(molecule_sizes) must equal len(T_A) = len(T_B)")
        
    T_A_with_indices = []
    T_B_with_indices = []
    
    current_pos = 0
    for i, size in enumerate(molecule_sizes):
        # Add index to all labels in this molecule's range
        for j in range(size):
            T_A_with_indices.append(f"{T_A[current_pos + j]}_{i}")
            T_B_with_indices.append(f"{T_B[current_pos + j]}_{i}")
        current_pos += size
        
    return np.array(T_A_with_indices, dtype=str), np.array(T_B_with_indices, dtype=str)


