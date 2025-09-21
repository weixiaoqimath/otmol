from typing import List, Tuple, Union, Optional
import numpy as np
import ot
from scipy.spatial import distance_matrix
import os

from .._optimal_transport import fsgw_mvc, perform_sOT_log
from ._distance_processing import geodesic_distance
from ._molecule_processing import write_xyz_with_custom_labels
from ._utils import is_permutation, permutation_to_matrix, cost_matrix, mismatched_bond_counter
from ._utils import root_mean_square_deviation, add_molecule_indices
from ._utils import add_perturbation, normalize_matrix, resolve_sinkhorn_conflicts


def molecule_alignment(
        X_A, 
        X_B, 
        T_A, 
        T_B, 
        B_A: np.ndarray = None,
        B_B: np.ndarray = None,
        method: str = 'fGW', 
        alpha_list: list = None, 
        molecule_sizes: List[int] = None,
        reflection: bool = False,
        cst_D: float = 0.,
        minimize_mismatched_edges: bool = False,
        save_path: str = None,
        return_BCI: bool = False,
        ) -> Tuple[np.ndarray, float, float]:
    """Compute alignment between two molecules or (molecular complexes) with optimal transport.

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
        List of alpha values to try for fGW or fsGW solver, by default None.
    molecule_sizes : List[int], optional
        Sizes of molecules, by default None. 
        It is only used when two structures contain multiple molecules,
        and molecules are ordered in the same way.
    reflection : bool, optional
        Whether to allow reflection in the Kabsch algorithm, by default False.
    cst_D : float, optional
        D = (1-cst_D)*Euclidean + cst_D*Geodesic, by default 0. If the user wants to reduce bond inconsistency, set cst_D to a value close to 1.
    minimize_mismatched_edges : bool, optional
        Whether to prioritize minimizing mismatched edges in the alignment, by default False.
    save_path : str, optional
        Path to save the aligned molecule, by default None. The atoms in the aligned molecule will be reordered.
    return_BCI: bool, optional
        Whether to return the BCI value (in range [0, 1]), by default False. Only use when minimize_mismatched_edges is False.

    Returns
    -------
    assignment : numpy.ndarray
        Optimal assignment between molecules.
    rmsd : float
        Best RMSD value.
    alpha : float
        Best alpha value.
    BCI : float
        BCI value. If minimize_mismatched_edges or return_BCI is True, the BCI value will be returned.
        A mismatched edge is an edge that is present in A but not in B.
        BCI is defined as the number of mismatched edges divided by the total number of edges in A.
    """
    if molecule_sizes is not None:
        T_B_original = T_B.copy()
        T_A, T_B = add_molecule_indices(T_A, T_B, molecule_sizes)
    C = cost_matrix(T_A = T_A, T_B = T_B, k = np.inf)

    C_finite = C.copy()
    C_finite[C_finite == np.inf] = 1e12
    if cst_D < 1e-5:
        D_A = distance_matrix(X_A, X_A)
        D_B = distance_matrix(X_B, X_B)
        D_A, D_B = D_A/D_A.max(), D_B/D_A.max()
    elif B_A is not None and B_B is not None:
        Euc_A, Euc_B = distance_matrix(X_A, X_A), distance_matrix(X_B, X_B)
        Geo_A, Geo_B = geodesic_distance(X_A, B_A), geodesic_distance(X_B, B_B)
        Euc_A, Euc_B = Euc_A/Euc_A.max(), Euc_B/Euc_A.max()
        if Geo_A.max() == np.inf:
            Geo_A[Geo_A == np.inf] = Euc_A[Geo_A == np.inf]
            Geo_B[Geo_B == np.inf] = Euc_B[Geo_B == np.inf]
        Geo_A, Geo_B = Geo_A/Geo_A.max(), Geo_B/Geo_A.max()
        D_A = (1-cst_D)*Euc_A + cst_D*Geo_A
        D_B = (1-cst_D)*Euc_B + cst_D*Geo_B
    rmsd_best = 1e10
    mismatched_bond_best = 1e10
    assignment_list = []
    assignment_set = set()
    assignment_best = None
    alpha_best = None
    _alpha_list = []
    X_B_aligned_best = None
    for alpha in alpha_list:
        if method == 'fGW':
            # Fused Gromov-Wasserstein
            P = ot.gromov.fused_gromov_wasserstein(C_finite, D_A, D_B, alpha=alpha, symmetric=True)
        elif method == 'fsGW':
            # Fused Supervised Gromov-Wasserstein
            P = fsgw_mvc(D_A, D_B, M=C, fsgw_alpha=alpha, fsgw_gamma=10, fsgw_niter=10, fsgw_eps=0.001)
        assignment = np.argmax(P, axis=1)
        if is_permutation(T_A=T_A, T_B=T_B, perm=assignment, case='single') and tuple(assignment) not in assignment_set:
            assignment_list.append(assignment)
            _alpha_list.append(alpha) # stores the alpha value for each assignment
            assignment_set.add(tuple(assignment))

    if minimize_mismatched_edges:  
        n = len(T_A)
        for i, assignment in enumerate(assignment_list):
            mismatched_bond = mismatched_bond_counter(B_A, B_B, assignment, n, n)
            if mismatched_bond < mismatched_bond_best:
                rmsd_best = 1e10 # reset rmsd_best
                mismatched_bond_best = mismatched_bond
                X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(assignment), reflection)
                rmsd = root_mean_square_deviation(X_A, X_B_aligned[assignment])
                if rmsd < rmsd_best:
                    rmsd_best = rmsd
                    assignment_best = assignment
                    alpha_best = _alpha_list[i]
                    X_B_aligned_best = X_B_aligned[assignment]
            if mismatched_bond == mismatched_bond_best:
                X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(assignment), reflection)
                rmsd = root_mean_square_deviation(X_A, X_B_aligned[assignment])
                if rmsd < rmsd_best:
                    rmsd_best = rmsd
                    assignment_best = assignment     
                    alpha_best = _alpha_list[i]
                    X_B_aligned_best = X_B_aligned[assignment]
        if assignment_best is None:
            print('No valid assignment found') 
            return None, None, None, None
        if save_path is not None:
            if molecule_sizes is not None:
                T_B = T_B_original
            write_xyz_with_custom_labels(
                os.path.join(save_path), 
                X_B_aligned_best, 
                T_B[assignment_best], 
                comment = 'aligned by OTMol'
                )
        BCI = mismatched_bond_counter(B_A, B_B, assignment_best, n, n, only_A_bonds=True)[0]/np.sum(B_A)*2
        return assignment_best, rmsd_best, alpha_best, BCI
    else:
        for i, assignment in enumerate(assignment_list):
            X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(assignment), reflection)
            rmsd = root_mean_square_deviation(X_A, X_B_aligned[assignment])
            if rmsd < rmsd_best:
                rmsd_best = rmsd
                assignment_best = assignment
                alpha_best = _alpha_list[i]
                X_B_aligned_best = X_B_aligned[assignment]
        if assignment_best is None:
            print('No valid assignment found')
            return None, None, None
        if save_path is not None:
            if molecule_sizes is not None:
                T_B = T_B_original
            write_xyz_with_custom_labels(
                os.path.join(save_path), 
                X_B_aligned_best, 
                T_B[assignment_best], 
                comment = 'aligned by OTMol'
                )
        if return_BCI:
            BCI = mismatched_bond_counter(B_A, B_B, assignment_best, len(T_A), len(T_B), only_A_bonds=True)[0]/np.sum(B_A)*2
            return assignment_best, rmsd_best, alpha_best, BCI
        else:
            return assignment_best, rmsd_best, alpha_best


def cluster_alignment(
        X_A: np.ndarray, 
        X_B: np.ndarray, 
        T_A: np.ndarray = None, 
        T_B: np.ndarray = None, 
        method: str = 'emd',
        p_list: list = None, 
        case: str = 'same element',
        reg: float = 1e-2, 
        numItermax: int = 1000, 
        n_atoms: int = None, 
        n_trials: int = 500,
        representative_option: str = 'center',
        reflection: bool = False,
        save_path: str = None,
        ):
    """Compute alignment between two clusters with optimal transport.

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
        Only used when case is 'same element'. List of power values for distance matrix, by default None.
    case : str
        Cluster type ('same element' or 'molecule cluster'), by default 'same element'.
    reg : float, optional
        Regularization parameter for sinkhorn and sOT, by default 1e-2.
    numItermax : int, optional
        Maximum number of iterations for sinkhorn and sOT, by default 1000.
    n_atoms : int, optional
        Number of atoms in a molecule in a molecule cluster, by default None.
    representative_option : str, optional
        The representative coordinate for a molecule in a molecule cluster, by default 'center'.
        For water clusters, one may choose "O" (oxygen).

    Returns
    -------
    assignment : numpy.ndarray
        Optimal assignment between clusters.
    rmsd : float
        Best RMSD value.
    p : float
        Best p value (for 'same element' case).
    """
    
    # Validate case argument
    valid_cases = ['same element', 'molecule cluster']
    if case not in valid_cases:
        raise ValueError(f"Invalid case '{case}'. Expected one of: {valid_cases}")
    
    rmsd_best = 1e10
    p_best = None
    _p_list = []
    permutation_best = None
    X_B_aligned_best = None
    assignment_list = []
    assignment_set = set()
    if case == 'same element':
        a = np.ones(X_A.shape[0])/X_A.shape[0]
        b = np.ones(X_B.shape[0])/X_B.shape[0]
        Euc_A, Euc_B = distance_matrix(X_A, X_A), distance_matrix(X_B, X_B)
        for p in p_list:
            D_A = Euc_A**p
            D_B = Euc_B**p
            D_A, D_B = D_A/D_A.max(), D_B/D_A.max()
            P = ot.gromov.gromov_wasserstein(D_A, D_B, symmetric=True)
            gw_assignment = np.argmax(P, axis=1)
            if is_permutation(perm=gw_assignment) and tuple(gw_assignment) not in assignment_set:   
                assignment_list.append(gw_assignment)
                _p_list.append(p)
                assignment_set.add(tuple(gw_assignment))
        for i, gw_assignment in enumerate(assignment_list):
            X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(gw_assignment), reflection)
            D_ot = distance_matrix(X_A, X_B_aligned)**2
            if method == 'emd':
                P_ot = ot.emd(a, b, D_ot/D_ot.max())
                ot_assignment = np.argmax(P_ot, axis=1)
                X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(ot_assignment), reflection)
                rmsd = root_mean_square_deviation(X_A, X_B_aligned[ot_assignment])
                if rmsd < rmsd_best:
                    rmsd_best = rmsd
                    p_best = _p_list[i]
                    permutation_best = ot_assignment
                    X_B_aligned_best = X_B_aligned[ot_assignment]
            if method == 'sinkhorn': # sinkhorn may not always output a permutation matrix
                P_ot = ot.sinkhorn(a, b, D_ot/D_ot.max(), reg=reg, numItermax=numItermax)
                assignments = resolve_sinkhorn_conflicts(P_ot)
                if assignments is None:
                    continue
                for ot_assignment in assignments:
                    X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(ot_assignment), reflection)
                    rmsd = root_mean_square_deviation(X_A, X_B_aligned[ot_assignment])
                    if rmsd < rmsd_best:
                        rmsd_best = rmsd
                        p_best = _p_list[i]
                        permutation_best = ot_assignment
                        X_B_aligned_best = X_B_aligned[ot_assignment]

        if permutation_best is None:
            print('No valid permutation found')
            return None, None, None
        if save_path is not None: # T_B will be used to write the xyz file
            write_xyz_with_custom_labels(
                os.path.join(save_path), 
                X_B_aligned_best, 
                T_B[permutation_best], 
                comment = 'aligned by OTMol'
                )
        return permutation_best, rmsd_best, p_best
    
    if case == 'molecule cluster':
        if representative_option == 'center':
            representative_A, representative_B = X_A.reshape(-1, n_atoms, 3).mean(axis=1), X_B.reshape(-1, n_atoms, 3).mean(axis=1)
        else:
            representative_A, representative_B = X_A[T_A == representative_option], X_B[T_B == representative_option]
        list_P = perturbation_before_gw(representative_A, representative_B, p_list = [1], n_trials = n_trials, scale = 0.1)
        print('The number of candidate molecular level permutations is', len(list_P))
        #c_A, c_B = np.sum(X_A, axis=0)/X_A.shape[0], np.sum(X_B, axis=0)/X_B.shape[0]
        for perm in list_P:
            P = permutation_to_matrix(perm)
            _, R, t = kabsch(representative_A, representative_B, P, reflection)
            X_B_aligned = (R @ X_B.T).T + t # or + c_A - R @ c_B, but seems no difference in the results
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
            ot_assignment = np.argmax(P_ot, axis=1) 
            X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(ot_assignment), reflection)
            rmsd = root_mean_square_deviation(X_A, X_B_aligned[ot_assignment])
            if rmsd < rmsd_best:
                rmsd_best = rmsd
                permutation_best = ot_assignment
                X_B_aligned_best = X_B_aligned[ot_assignment]

        if permutation_best is None:
            print('No valid permutation found')
            return None, None
        if save_path is not None: 
            write_xyz_with_custom_labels(
                os.path.join(save_path), 
                X_B_aligned_best, 
                T_B[permutation_best], 
                comment = 'aligned by OTMol'
                )
        return permutation_best, rmsd_best        

def kabsch(
    X1: np.ndarray,
    X2: np.ndarray,
    P: np.ndarray,
    reflection: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kabsch algorithm. 
    Perform rigid body rotation (including reflection if reflection is set to True), 
    and translation to align molecules.

    Parameters
    ----------
    X1 : numpy.ndarray
        Coordinates of molecule 1 (reference) as an n x 3 array.
    X2 : numpy.ndarray
        Coordinates of molecule 2 (to be aligned) as an m x 3 array.
    P : numpy.ndarray
        A permutation matrix describing the atom assignment between molecules, 
        where matrix[i, j] = 1 if assignment[i] = j.
    reflection : bool, optional
        Whether to allow reflection, by default False. 

    Returns
    -------
    X2_aligned : numpy.ndarray
        Aligned coordinates of molecule 2.
    R : numpy.ndarray
        Rotation matrix.
    t : numpy.ndarray
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
    
    if not reflection and np.linalg.det(U @ Vt) < 0: # Ensure R is a proper rotation matrix (det(R)=1)
        Vt[-1, :] *= -1
        R = U @ Vt
    else:
        R = U @ Vt
        
    # Compute the translation
    t = mu1 - R @ mu2
    
    # Transform X2
    X2_aligned = (R @ X2.T).T + t
    
    return X2_aligned, R, t


def perturbation_before_gw(
        X_A: np.ndarray, 
        X_B: np.ndarray, 
        p_list: list = [1], 
        n_trials: int = 100, 
        scale: float = 0.1, 
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
    p_list : list, 
        Power of the distance matrix, by default [1].
    n_trials : int, optional
        Number of trials to run, by default 100.
    scale : float, optional
        Standard deviation of the Gaussian noise, by default 0.1.

    Returns
    -------
    list_perms : List[numpy.ndarray]
        List of permutations.
    """
    unique_perms = set()
    list_perms = []
    # It seems that when the number of atoms is 2, the GW algorithm always returns the permutation [0,1] regardless of the input.
    # So we handle this case separately.
    if len(X_A) == 2:
        list_perms = [np.array([0,1]), np.array([1,0])]
        return list_perms
  
    for i in range(n_trials):
        X_A_perturbed, X_B_perturbed = add_perturbation(X_A, scale, random_state = i), add_perturbation(X_B, scale, random_state = i)
        Euc_A, Euc_B = distance_matrix(X_A_perturbed, X_A_perturbed), distance_matrix(X_B_perturbed, X_B_perturbed)
        for p in p_list:
            D_A = Euc_A**p
            D_B = Euc_B**p
            D_A, D_B = D_A/D_A.max(), D_B/D_A.max()
            P = ot.gromov.gromov_wasserstein(D_A, D_B, symmetric=True)
            perm = np.argmax(P, axis=1)
            if not is_permutation(perm=perm):
                continue
            X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(perm), reflection=False)
            D_ot = distance_matrix(X_A, X_B_aligned)**2
            P = ot.emd([], [], D_ot/D_ot.max())
            perm = np.argmax(P, axis=1)
            if not is_permutation(perm=perm):
                continue
            perm_tuple = tuple(perm)
            if perm_tuple not in unique_perms:
                unique_perms.add(perm_tuple)
                list_perms.append(perm)                
    return list_perms


#def molecule_alignment_with_perturbation(
#        X_A, 
#        X_B, 
#        T_A, 
#        T_B, 
#        B_A: np.ndarray = None,
#        B_B: np.ndarray = None,
#        alpha_list: list = np.arange(0,1,0.1)[1:], 
#        molecule_sizes: List[int] = None,
#        reflection: bool = False,
#        scale: float = 0.1,
#        #cst_D: float = 0.,
#        n_perturbation: int = 30,
#        ) -> Tuple[np.ndarray, float, float]:
#    """Compute optimal transport and alignment between molecules.

#    Parameters
#    ----------
#    X_A : numpy.ndarray
#        Coordinates of molecule A.
#    X_B : numpy.ndarray
#        Coordinates of molecule B.
#    T_A : array_like
#        Atom labels of molecule A.
#    T_B : array_like
#        Atom labels of molecule B.
#    method : list of str
#        Optimal transport method to use, by default ['fgw', 'emd'].
#    alpha_list : list
#        List of alpha values to try for fGW or fsGW, by default None.
#    molecule_sizes : List[int], optional
#        Sizes of molecules, by default None.
#    reg : float, optional
#        Regularization parameter for sinkhorn, by default 1e-2.

#    Returns
#    -------
#    numpy.ndarray
#        Optimal assignment between molecules.
#    float
#        Best RMSD value.
#    float
#        Best alpha value.
#    """
#    if molecule_sizes is not None:
#        T_A, T_B = add_molecule_indices(T_A, T_B, molecule_sizes)
#    C = cost_matrix(T_A = T_A, T_B = T_B, k = np.inf)

#    C_finite = C.copy()
#    C_finite[C_finite == np.inf] = 1e12
#    D_A = geodesic_distance(X_A, B_A)
#    D_B = geodesic_distance(X_B, B_B)
#    D_A, D_B = D_A/D_A.max(), D_B/D_A.max()
#    rmsd_best = 1e10
#    mismatched_bond_best = 1e10
#    assignment_list = []
#    assignment_set = set()
#    assignment_best = None
#    alpha_best = None
#    for alpha in alpha_list:
#        P = ot.gromov.fused_gromov_wasserstein(C_finite, D_A, D_B, alpha=alpha, symmetric=True)
#        assignment = np.argmax(P, axis=1)
#        if is_permutation(T_A=T_A, T_B=T_B, perm=assignment, case='single') and tuple(assignment) not in assignment_set:
#            assignment_list.append(assignment)
#            assignment_set.add(tuple(assignment))

#    for i in range(n_perturbation):
#        X_A_perturbed = add_perturbation(X = X_A, noise_scale = scale, random_state = i) 
#        X_B_perturbed = add_perturbation(X = X_B, noise_scale = scale, random_state = i) 
#        D_A_perturbed = geodesic_distance(X_A_perturbed, B_A)
#        D_B_perturbed = geodesic_distance(X_B_perturbed, B_B)
#        D_A_perturbed, D_B_perturbed = D_A_perturbed/D_A_perturbed.max(), D_B_perturbed/D_A_perturbed.max() 
#        for j in alpha_list:
#            P = ot.gromov.fused_gromov_wasserstein(C, D_A_perturbed, D_B_perturbed, alpha = j, symmetric=True)
#            assignment = np.argmax(P, axis=1)
#            if is_permutation(T_A, T_B, assignment, case='single') and tuple(assignment) not in assignment_set:
#                assignment_list.append(assignment)  
#                assignment_set.add(tuple(assignment))

#    n = len(T_A)
#    for assignment in assignment_list:
#        mismatched_bond = mismatched_bond_counter(B_A, B_B, assignment, n, n)
#        if mismatched_bond < mismatched_bond_best:
#            rmsd_best = 1e10 # reset rmsd_best
#            mismatched_bond_best = mismatched_bond
#            X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(assignment), reflection)
#            rmsd = root_mean_square_deviation(X_A, X_B_aligned[assignment])
#            if rmsd < rmsd_best:
#                rmsd_best = rmsd
#                assignment_best = assignment
#        if mismatched_bond == mismatched_bond_best:
#            X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(assignment), reflection)
#            rmsd = root_mean_square_deviation(X_A, X_B_aligned[assignment])
#            if rmsd < rmsd_best:
#                rmsd_best = rmsd
#                assignment_best = assignment              
#    if assignment_best is None:
#         print('No valid assignment found') 
#    return assignment_best, rmsd_best, alpha_best, mismatched_bond_best

