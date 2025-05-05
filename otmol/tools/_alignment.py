from .._optimal_transport import fsgw_mvc, perform_sOT_log
from .._optimal_transport import supervised_gromov_wasserstein
#from .._optimal_transport import supervised_optimal_transport
from ._utils import is_permutation, permutation_to_matrix, cost_matrix
from ._utils import root_mean_square_deviation 
from ._utils import add_perturbation, normalize_matrix, resolve_sinkhorn_conflicts
import ot
import numpy as np
from typing import List, Tuple, Union, Optional
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt


def molecule_ot_and_alignment(
        X_A, 
        X_B, 
        T_A, 
        T_B, 
        method: List[str] = ['fGW', 'emd'], 
        alpha_list: list = None, 
        case: str = 'single', 
        multiple_molecules_block_size: List[int] = None,
        reg: float = 1e-2,
        ) -> Tuple[np.ndarray, float, float]:
    """Compute optimal transport and alignment between molecules.

    Parameters
    ----------
    X_A : numpy.ndarray
        Coordinates of molecule A.
    X_B : numpy.ndarray
        Coordinates of molecule B.
    T_A : array_like
        Atom labels of molecule A.
    T_B : array_like
        Atom labels of molecule B.
    method : list of str
        Optimal transport method to use, by default ['fgw', 'emd'].
    alpha_list : list
        List of alpha values to try for fGW or fsGW, by default None.
    case : str
        Case type ('single' or 'multiple'), by default 'single'.
    multiple_molecules_block_size : List[int], optional
        Block sizes for multiple molecules case, by default None.
    reg : float, optional
        Regularization parameter for sinkhorn, by default 1e-2.

    Returns
    -------
    numpy.ndarray
        Optimal assignment between molecules.
    float
        Best RMSD value.
    float
        Best alpha value.
    """
    if case == 'single': 
        C = cost_matrix(T_A = T_A, T_B = T_B, k = np.inf)
        C = normalize_matrix(C)
    if case == 'multiple':
        if multiple_molecules_block_size is None:
            raise ValueError('multiple_molecules_block_size is required for multiple molecules')
        C = cost_matrix(T_A = T_A, T_B = T_B, k = np.inf, multiple_molecules_block_size = multiple_molecules_block_size)
        C = normalize_matrix(C)
    C_finite = C.copy()
    C_finite[C_finite == np.inf] = 1e12
    D_A = distance_matrix(X_A, X_A)
    D_B = distance_matrix(X_B, X_B)
    rmsd_best = 1e10
    permutation_best = None
    alpha_best = None
    for alpha in alpha_list:
        if method[0] == 'fGW':
            # Fused Gromov-Wasserstein
            P = ot.gromov.fused_gromov_wasserstein(C_finite, D_A/D_A.max(), D_B/D_A.max(), alpha=alpha, symmetric=True)
        elif method[0] == 'fsGW':
            # Fused Supervised Gromov-Wasserstein
            P = fsgw_mvc(D_A/D_A.max(), D_B/D_A.max(), M=C, fsgw_alpha=alpha, fsgw_gamma=10, fsgw_niter=10, fsgw_eps=0.001)
        assignment = np.argmax(P, axis=1)
        if not is_permutation(T_A=T_A, T_B=T_B, perm=assignment, case=case, multiple_molecules_block_size=multiple_molecules_block_size):
            continue

        # OT
        X_B_aligned, _, _ = molecule_alignment_allow_reflection(X_A, X_B, permutation_to_matrix(assignment))
        D_ot = distance_matrix(X_A, X_B_aligned)**2
        D_ot[C == np.inf] = np.inf
        #mask = np.isfinite(D_ot)
        #print(np.max(D_ot[mask]))
        D_ot = normalize_matrix(D_ot)
        a = np.ones(X_A.shape[0])/X_A.shape[0]
        b = np.ones(X_B.shape[0])/X_B.shape[0]
        if method[1] == 'emd':
            D_ot[D_ot == np.inf] = 1e12
            P_ot = ot.emd(a, b, D_ot)
        if method[1] == 'sinkhorn':
            D_ot[D_ot == np.inf] = 1e12
            P_ot = ot.sinkhorn(a, b, D_ot, reg=reg)
        if method[1] == 'sOT':
            options = {
                'niter_sOT': 10**3,
                'f_init': np.zeros(X_A.shape[0]),
                'g_init': np.zeros(X_B.shape[0]),
                'penalty': 10,
                'stopthr': 1e-8,
            }
            P_ot, _, _ = perform_sOT_log(D_ot, a, b, 1e-3, options)
        if not is_permutation(T_A=T_A, T_B=T_B, perm=np.argmax(P_ot, axis=1), case=case, multiple_molecules_block_size=multiple_molecules_block_size):
            continue
        rmsd = root_mean_square_deviation(X_A, X_B_aligned[np.argmax(P_ot, axis=1)])
        if rmsd < rmsd_best:
            rmsd_best = rmsd
            permutation_best = np.argmax(P_ot, axis=1)
            alpha_best = alpha
    if permutation_best is None:
        raise ValueError('No valid permutation found')
    return permutation_best, rmsd_best, alpha_best


def cluster_ot_and_alignment(
        X_A: np.ndarray, 
        X_B: np.ndarray, 
        T_A: np.ndarray = None, 
        T_B: np.ndarray = None, 
        method: str = 'emd',
        p_list: list = None, 
        case: str = 'same elements',
        reg: float = 1e-2, 
        numItermax: int = 1000, 
        n_atoms: int = None, # for molecule cluster, the number of atoms in each molecule
        molecule_cluster_options: str = 'center',
        ):
    """Compute optimal transport and alignment between clusters.

    Parameters
    ----------
    X_A : numpy.ndarray
        Coordinates of cluster A.
    X_B : numpy.ndarray
        Coordinates of cluster B.
    T_A : numpy.ndarray, optional
        Atom labels of cluster A, by default None.
    T_B : numpy.ndarray, optional
        Atom labels of cluster B, by default None.
    method : str, optional
        Method to use for optimal transport, by default 'emd'.
    p_list : list, optional
        Only used when case is 'same elements'. List of power values for distance matrix, by default None.
    case : str
        Case type ('same elements' or 'water cluster'), by default 'same elements'.
    reg : float, optional
        Regularization parameter for sinkhorn and sOT, by default 1e-2.
    numItermax : int, optional
        Maximum number of iterations for sinkhorn and sOT, by default 1000.
    n_atoms : int, optional
        Number of atoms in a molecule in a molecule cluster, by default None.
    molecule_cluster_options : str, optional
        Options for molecule cluster, by default 'center'.

    Returns
    -------
    numpy.ndarray
        Optimal assignment between clusters.
    float
        Best RMSD value.
    float
        Best p value (for 'same elements' case).
    """
    if case == 'same elements':
        rmsd_best = 1e10
        p_best = None
        permutation_best = None
        for p in p_list:
            D_A = distance_matrix(X_A, X_A)**p
            D_B = distance_matrix(X_B, X_B)**p
            P = ot.gromov.gromov_wasserstein(D_A/D_A.max(), D_B/D_A.max())
            gw_assignment = np.argmax(P, axis=1)
            if is_permutation(perm=gw_assignment):   
                X_B_aligned, _, _ = molecule_alignment_allow_reflection(X_A, X_B, permutation_to_matrix(gw_assignment))
                a = np.ones(X_A.shape[0])/X_A.shape[0]
                b = np.ones(X_B.shape[0])/X_B.shape[0]
                if method == 'emd':
                    P_ot = ot.emd(a, b, normalize_matrix(distance_matrix(X_A, X_B_aligned)**2))
                    ot_assignment = np.argmax(P_ot, axis=1)
                if method == 'sinkhorn': # sinkhorn may not always output a permutation matrix
                    P_ot = ot.sinkhorn(a, b, normalize_matrix(distance_matrix(X_A, X_B_aligned)**2), reg=reg, numItermax=numItermax)
                    ot_assignment = resolve_sinkhorn_conflicts(P_ot)
                X_B_aligned, _, _ = molecule_alignment_allow_reflection(X_A, X_B, permutation_to_matrix(ot_assignment))
                rmsd = root_mean_square_deviation(X_A, X_B_aligned[ot_assignment])
                if rmsd < rmsd_best:
                    rmsd_best = rmsd
                    p_best = p
                    permutation_best = ot_assignment
        if permutation_best is None:
            raise ValueError('No valid permutation found')
        return permutation_best, rmsd_best, p_best
    
    if case == 'molecule cluster':
        if molecule_cluster_options == 'center':
            representative_A, representative_B = X_A.reshape(-1, n_atoms, 3).mean(axis=1), X_B.reshape(-1, n_atoms, 3).mean(axis=1)
        else:
            representative_A, representative_B = X_A[T_A == molecule_cluster_options], X_B[T_B == molecule_cluster_options]
        list_P = perturbation_before_gw(representative_A, representative_B, p = 1, n_trials = 500, scale = 0.1)
        print('The number of candidate center permutations is', len(list_P))
        rmsd_best = 1e10
        permutation_best = None
        for perm in list_P:
            P = permutation_to_matrix(perm)
            _, R, t = molecule_alignment_allow_reflection(representative_A, representative_B, P)
            X_B_aligned = (R @ X_B.T).T + t
            a, b = np.ones(X_A.shape[0])/X_A.shape[0], np.ones(X_B.shape[0])/X_B.shape[0]
            # construct a distance matrix such that one water molecule is mapped to another according to P
            D_ot = np.full((X_A.shape[0], X_B.shape[0]), np.inf)
            for i in range(X_A.shape[0]//n_atoms): 
                j = np.argmax(P[i])
                X_A_i, X_B_aligned_j = X_A[i*n_atoms:(i+1)*n_atoms], X_B_aligned[j*n_atoms:(j+1)*n_atoms]
                T_A_i, T_B_j = T_A[i*n_atoms:(i+1)*n_atoms], T_B[j*n_atoms:(j+1)*n_atoms]
                D_ot[i*n_atoms:(i+1)*n_atoms, j*n_atoms:(j+1)*n_atoms] = cost_matrix(X_A_i, X_B_aligned_j, T_A_i, T_B_j, np.inf)

            D_ot = normalize_matrix(D_ot)
            if method == 'emd':
                D_ot[D_ot == np.inf] = 1e12
                P_ot = ot.emd(a, b, D_ot)
            if method == 'sOT':
                options = {
                    'niter_sOT': numItermax,
                    'f_init': np.zeros(X_A.shape[0]),
                    'g_init': np.zeros(X_B.shape[0]),
                    'penalty': 10,
                    'stopthr': 1e-8
                }
                P_ot, _, _ = perform_sOT_log(D_ot, a, b, reg, options) 
                if not is_permutation(T_A, T_B, np.argmax(P_ot, axis=1), case = 'molecule cluster', n_atoms = n_atoms):
                    continue
            X_B_aligned, _, _ = molecule_alignment_allow_reflection(X_A, X_B, P_ot)
            rmsd = root_mean_square_deviation(X_A, X_B_aligned[np.argmax(P_ot, axis=1)])

            if rmsd < rmsd_best:
                rmsd_best = rmsd
                permutation_best = np.argmax(P_ot, axis=1)

        return permutation_best, rmsd_best        

def molecule_alignment_allow_reflection(
    X1: np.ndarray,
    X2: np.ndarray,
    P: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kabsch algorithm. Perform rigid body rotation (including reflection), and translation to align molecules.

    Parameters
    ----------
    X1 : numpy.ndarray
        Coordinates of molecule 1 (template) as an n x 3 array.
    X2 : numpy.ndarray
        Coordinates of molecule 2 (to be aligned) as an m x 3 array.
    P : numpy.ndarray
        A transport plan describing the correspondence between molecules. 
        In the study of isomers, it should be a permutation matrix.

    Returns
    -------
    numpy.ndarray
        Aligned coordinates of molecule 2.
    numpy.ndarray
        Rotation matrix.
    numpy.ndarray
        Translation vector.
    """
    total_weight = P.sum()
    
    # Compute weights for each point
    w1 = P.sum(axis=1)  # weights for X1
    w2 = P.sum(axis=0)  # weights for X2
    
    # Compute weighted centroids
    mu1 = np.sum(X1 * w1[:, None], axis=0) / total_weight
    mu2 = np.sum(X2 * w2[:, None], axis=0) / total_weight
    
    # Center the point clouds
    X1_centered = X1 - mu1
    X2_centered = X2 - mu2
    
    # vectorized calculation: H = X1_centered^T P X2_centered 
    H = X1_centered.T @ P @ X2_centered
    
    # Compute SVD of H
    U, _, Vt = np.linalg.svd(H)
    
    # Allow reflection by not enforcing det(R)==1
    R = U @ Vt
    
    # Compute the translation
    t = mu1 - R @ mu2
    
    # Transform X2
    X2_aligned = (R @ X2.T).T + t
    
    return X2_aligned, R, t

def molecule_alignment_no_reflection(
    X1: np.ndarray,
    X2: np.ndarray,
    P: np.ndarray
):
    """
    Perform rigid body rotation and translation to align the second point cloud onto the first point cloud guided by an OT plan.

    Parameters
    ----------
    X1 : numpy.ndarray
        Coordinates of molecule 1 (template) as an n x 3 array.
    X2 : numpy.ndarray
        Coordinates of molecule 2 (to be aligned) as an m x 3 array.
    P : numpy.ndarray
        Optimal transport plan describing the correspondence between molecules.

    Returns
    -------
    numpy.ndarray
        Aligned coordinates of molecule 2.
    numpy.ndarray
        Rotation matrix.
    numpy.ndarray
        Translation vector.
    """
    total_weight = P.sum()
    
    # Compute weights for each point in X1 and X2
    w1 = P.sum(axis=1)  # n-dimensional: weight for each row of X1
    w2 = P.sum(axis=0)  # m-dimensional: weight for each row of X2
    
    # Compute weighted centroids
    mu1 = np.sum(X1 * w1[:, None], axis=0) / total_weight
    mu2 = np.sum(X2 * w2[:, None], axis=0) / total_weight
    
    # Center the point clouds
    X1_centered = X1 - mu1
    X2_centered = X2 - mu2
    
    # Compute the weighted cross-covariance matrix H
    #d = X1.shape[1]  # dimensionality, should be 3
    #H = np.zeros((d, d))
    #n, m = P.shape
    #for i in range(n):
    #    for j in range(m):
    #        H += P[i, j] * np.outer(X1_centered[i], X2_centered[j])
    
    # Alternatively, a vectorized version could be used if desired.
    H = (X1_centered.T @ P) @ X2_centered
    
    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    
    # Ensure R is a proper rotation matrix (det(R)=1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    
    # Compute the translation
    t = mu1 - R @ mu2
    
    # Apply the transformation to X2
    X2_aligned = (R @ X2.T).T + t
    
    return X2_aligned, R, t

def perturbation_before_gw(
        X_A: np.ndarray, 
        X_B: np.ndarray, 
        p: float = 1, 
        n_trials: int = 100, 
        scale: float = 0.1, 
        threshold: float = None
        ) -> List[np.ndarray]:
    """Find various suboptimal transport plans between clusters of atoms.

    When calculating the distance matrix, Gaussian noise is added to the coordinates
    to generate various suboptimal transport plans.
    We first do a GW, then do a Kantorovich (ot.emd) on the aligned coordinates from GW.

    Parameters
    ----------
    X_A : numpy.ndarray
        Coordinates of cluster A.
    X_B : numpy.ndarray
        Coordinates of cluster B.
    p : float, optional
        Power of the distance matrix, by default 1.
    n_trials : int, optional
        Number of trials to run, by default 100.
    scale : float, optional
        Standard deviation of the Gaussian noise, by default 0.1.
    threshold : float, optional
        RMSD threshold for accepting a permutation, by default None.

    Returns
    -------
    List[numpy.ndarray]
        List of permutations.
    """
    unique_perms = set()
    list_perms = []

    # It seems that when the number of atoms is 2, the GW algorithm always returns the permutation [0,1] regardless of the input.
    # So we handle this case separately.
    if len(X_A) == 2:
        return [np.array([0,1]), np.array([1,0])]
              
    for i in range(n_trials):
        X_A_perturbed, X_B_perturbed = add_perturbation(X_A, scale, random_state = i), add_perturbation(X_B, scale, random_state = i)
        D_A = distance_matrix(X_A_perturbed, X_A_perturbed)**p
        D_B = distance_matrix(X_B_perturbed, X_B_perturbed)**p
        P_gw = ot.gromov.gromov_wasserstein(D_A/D_A.max(), D_B/D_A.max(), symmetric=True)
        if len(np.unique(np.argmax(P_gw, axis=1))) == P_gw.shape[0]:
            X_B_aligned, _, _ = molecule_alignment_allow_reflection(X_A, X_B, permutation_to_matrix(np.argmax(P_gw, axis=1)))
            D_ot = distance_matrix(X_A, X_B_aligned)**2
            P_ot = ot.emd([], [], normalize_matrix(D_ot))
            X_B_aligned, _, _ = molecule_alignment_allow_reflection(X_A, X_B, permutation_to_matrix(np.argmax(P_ot, axis=1)))
            if threshold is None:
                perm = np.argmax(P_ot, axis=1)
                # Check for uniqueness
                perm_tuple = tuple(perm)
                if perm_tuple not in unique_perms:
                    unique_perms.add(perm_tuple)
                    list_perms.append(perm)                
            elif root_mean_square_deviation(X_A, X_B_aligned[np.argmax(P_ot, axis=1)]) < threshold:
                perm = np.argmax(P_ot, axis=1)
                # Check for uniqueness
                perm_tuple = tuple(perm)
                if perm_tuple not in unique_perms:
                    unique_perms.add(perm_tuple)
                    list_perms.append(perm)   
    return list_perms

