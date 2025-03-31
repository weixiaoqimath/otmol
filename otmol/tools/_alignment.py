from .._optimal_transport import fused_supervised_gromov_wasserstein
from .._optimal_transport import supervised_gromov_wasserstein
from .._optimal_transport import supervised_optimal_transport

import ot
import numpy as np

def molecule_optimal_transport(
    C: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    method: str = 'fgw',
    alpha: float = 0.5,
    gamma: float = 2.0,
    eps: float = 0.1,
    fsgw_gamma: float = 2.0,
    fsgw_niter: int = 10,
    sgw_gamma: float = 2.0,
    sgw_niter: int = 20,
    sgw_cutoff: float = np.inf
):
    """
    Compute optimal transport plan between molecules, each with n and m atoms.

    Parameters
    ----------
    C
        The cost matrix of shape ``n`` × ``m`` between atoms from the two molecules, such as distance in physical chemical properties.
    D1
        An ``n`` x ``n`` distance matrix between atoms in molecule 1. Examples include (1) geodesic distance on the graph with edges representing bonds and (2) simply Euclidean distance.
    D2
        Similar to ``D1``, but an ``m`` x ``m`` matrix for molecule 2.
    method
        The optimal transport method to use. 
        `'fgw`' fused Gromov-Wasserstein,
        `'efgw`' entropy regularized fused Gromov-Wasserstein,
        `'fugw`' fused unbalanced Gromov-Wasserstein,
        `'fsgw`' fused supervised Gromov-Wasserstein,
        `'gw`' Gromov-Wasserstein,
        `'egw`' entropy regularized Gromov-Wasserstein,
        `'ugw`' unbalanced Gromov-Wasserstein,
        `'sgw`' supervised Gromov-Wasserstein.
    alpha
        The weight for the GW term if fused GW is used. The weight for the Wasserstein term will be (1-alpha).
    gamma
        The coefficient for KL divergence if unbalanced GW or fused unbalanced GW is used.
    eps
        The coefficient for the entropy regularization term.
    fsgw_gamma
        The coefficient for the penalty term of untransported mass in fused supervised GW.
    fsgw_niter
        The number of iterations in the fsgw algorithm.
    sgw_gamma
        The coefficient for the penalty term or the untransported mass in supervised GW.
    sgw_niter
        The number of iterations in the sgw algorithm.
    sgw_cutoff
        The gw cutoff value if sgw or fsgw is used.

    Returns
    -------
    P : np.ndarray
        The ``n`` x ``m`` optimal transport matrix between the two molecules.
    """

    if method == 'fgw':
        P = ot.gromov.fused_gromov_wasserstein(C, D1, D2, alpha=alpha)
    elif method == 'efgw':
        P = ot.gromov.entropic_fused_gromov_wasserstein(C, D1, D2, alpha=alpha, epsilon=eps)
    elif method == 'fugw':
        P,_ = ot.gromov.fused_unbalanced_gromov_wasserstein(D1, D2, M=C, reg_marginals=gamma, epsilon=eps, alpha=(1-alpha)/alpha)
    elif method == 'fsgw':
        P = fused_supervised_gromov_wasserstein(D1, D2, C, fsgw_niter=fsgw_niter, fsgw_eps=eps, fsgw_alpha=alpha, fsgw_gamma=fsgw_gamma, gw_cutoff=sgw_cutoff)
    elif method == 'gw':
        P = ot.gromov.fused_gromov_wasserstein(C, D1, D2, alpha=alpha)
    elif method == 'egw':
        P = ot.gromov.entropic_gromov_wasserstein(C, D1, D2, alpha=alpha, epsilon=eps)
    elif method == 'ugw':
        P,_ = ot.gromov.fused_unbalanced_gromov_wasserstein(D1, D2, M=C, reg_marginals=gamma, epsilon=eps, alpha=0)
    elif method == 'sgw':
        P = supervised_gromov_wasserstein(D1, D2, eps=eps, nitermax=sgw_niter, threshold=sgw_cutoff)
    
    return P

def molecule_alignment_allow_reflection(
    X1: np.ndarray,
    X2: np.ndarray,
    P: np.ndarray
):
    """
    Perform rigid body rotation, reflection, and translation to align the second point cloud onto the first point cloud guided by an OT plan.

    Parameters
    ----------
    X1
        The 3D coordinates of molecule 1 (template) as an ``n`` x ``3`` array.
    X2
        The 3D coordinates of molecule 2 (to be aligned) as an ``m`` x ``3`` array.
    P
        The ``n`` x ``m`` optimal transport plan describing the correspondence between the molecules.

    Returns
    -------
    X2_aligned : np.ndarray
        The ``m`` x ``3`` coordinates matrix of the aligned molecule 2.
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
    
    # Compute weighted cross-covariance matrix
    d = X1.shape[1]  # should be 3
    H = np.zeros((d, d))
    n, m = P.shape
    for i in range(n):
        for j in range(m):
            H += P[i, j] * np.outer(X1_centered[i], X2_centered[j])
    
    # Compute SVD of H
    U, _, Vt = np.linalg.svd(H)
    
    # Allow reflection by not enforcing det(R)==1
    R = U @ Vt
    
    # Compute the translation
    t = mu1 - R @ mu2
    
    # Transform X2
    X2_aligned = (R @ X2.T).T + t
    
    return X2_aligned

def molecule_alignment_no_reflection(
    X1: np.ndarray,
    X2: np.ndarray,
    P: np.ndarray
):
    """
    Perform rigid body rotation and translation to align the second point cloud onto the first point cloud guided by an OT plan.

    Parameters
    ----------
    X1
        The 3D coordinates of molecule 1 (template) as an ``n`` x ``3`` array.
    X2
        The 3D coordinates of molecule 2 (to be aligned) as an ``m`` x ``3`` array.
    P
        The ``n`` x ``m`` optimal transport plan describing the correspondence between the molecules.

    Returns
    -------
    X2_aligned : np.ndarray
        The ``m`` x ``3`` coordinates matrix of the aligned molecule 2.
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
    d = X1.shape[1]  # dimensionality, should be 3
    H = np.zeros((d, d))
    n, m = P.shape
    for i in range(n):
        for j in range(m):
            H += P[i, j] * np.outer(X1_centered[i], X2_centered[j])
    
    # Alternatively, a vectorized version could be used if desired.
    
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
    
    return X2_aligned