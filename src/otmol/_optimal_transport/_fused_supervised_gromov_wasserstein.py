# fsgw matching
# Modified from Yaqi Wu's code. 
# only used if fsGW solver is used

import numpy as np
import itertools
import random
from scipy import sparse
import networkx as nx
from tqdm import tqdm
from scipy.spatial import distance
import scipy
import anndata as ad
import scanpy as sc
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse

######################################################################

####        indices order of D1, D2 and C should be consistent   ####
####        feature cost M should be normalized as input of fsgw ####
####        should pick cutoff_GW in ( 0, max(D1,D2) )           ####
####        should pick cutoff_CC in (0,1)                       ####

######################################################################


#example code:

# from tqdm import tqdm


# cutoff_GW_list = [20, 30]
# cutoff_CC_list = [0.5, 0.6]

# for cutoff_GW in tqdm(cutoff_GW_list, desc="Running different GW CC cutoffs"):
#     for cutoff_CC in tqdm(cutoff_CC_list, leave=False):
#         tqdm.write(f"\nRunning GW cutoff = {cutoff_GW}, CC cutoff = {cutoff_CC}")
#         P = _fsgw_align.fsgw_mvc(D_A_test, D_B_test, M, gw_cutoff=cutoff_GW, w_cutoff=cutoff_CC)
#         tqdm.write(f"P.sum(): {P.sum()}")
#         save_path = os.path.join(save_folder, f"ds_matching_{pair}_{cutoff_GW}_{cutoff_CC}.csv")
#         np.savetxt(save_path, P, delimiter=",", fmt='%.8f')
#         tqdm.write(f"Saved to: {save_path}")

###################################




def matrix_rescaling_checking(C_norm, verbose=True, atol=1e-12):
    min_val = C_norm.min()
    max_val = C_norm.max()
    if not (min_val >= -atol and max_val <= 1.0 + atol):
        raise AssertionError("Rescaling failed: data not in [0,1] interval.")

    if not np.isclose(max_val, 1.0, atol=atol):
        raise AssertionError("Rescaling failed: max is not close to 1.")

    if verbose:
        print(f"After rescaling: min = {min_val}, max = {max_val}.")


def fsgw_mvc(
    D1,
    D2,
    M, # range (0,1)
    gw_cutoff = np.inf,
    w_cutoff = np.inf,
    fsgw_niter = 10,
    fsgw_eps = 0.01,
    fsgw_alpha = 0.1,
    fsgw_gamma = 2,
    
):
    n = D1.shape[0]
    m = D2.shape[0]
 
    t = D1[D1 > 0].max()
    D1_norm = D1 / t
    D2_norm = D2 / t
    
    #P_idx = np.array( [[i, j] for i, j in itertools.product(range(n), range(m))], dtype=int )
    
    #I = []
    #J = []
    #for i in range(len(P_idx)):
    #    D_tmp = (D1[P_idx[i][0],:,np.newaxis] - D2[P_idx[i][1],:])**2
    #    tmp_idx = np.where(D_tmp.flatten() <= gw_cutoff**2)[0]
    #    J.extend(list(tmp_idx))
    #    I.extend([i for _ in range(len(tmp_idx))])
    #I = np.array(I, int)
    #J = np.array(J, int)
    #D = np.ones_like(I)
    #A = sparse.coo_matrix((D, (I, J)), shape=(n*m, n*m))
    
    
    #G = nx.from_scipy_sparse_array(A)
    #tmp_idx = np.where(M.flatten() >  w_cutoff)[0]
    #G.remove_nodes_from(tmp_idx)

    
    #M_flatten = M.flatten()
    #zero_indices = set()
    #G_copy = nx.complement(G)
    #del G
    #with tqdm(total=G_copy.number_of_edges(), desc=f"Finding min vertex covering for cutoff_GW {gw_cutoff} and cutoff_CC {w_cutoff}") as pbar:
    #    while G_copy.edges:
    #        initial_edges = G_copy.number_of_edges()
    #        max_degree_vertices, max_degree = vertex_with_most_edges(G_copy)
    #        max_Cij_value = -float('inf')
    #        vertex_with_max_Cij = None
    #        for vertex in max_degree_vertices:
    #            if M_flatten[vertex] > max_Cij_value:
    #                max_Cij_value = M_flatten[vertex]
    #                vertex_with_max_Cij = vertex
    
    #        G_copy.remove_node(vertex_with_max_Cij)
    #        zero_indices.add(vertex_with_max_Cij)
    #        removed_edges = initial_edges - G_copy.number_of_edges()
    #        pbar.update(removed_edges)
    #del G_copy

    
    #zero_indices = list(tmp_idx)
    #zero_indices = np.array(zero_indices)
    #print("# of potential non-zeros in P:", n*m - len(zero_indices))

    
    #row_idx = P_idx[zero_indices,0]
    #col_idx = P_idx[zero_indices,1]
    row_idx, col_idx = np.where(np.isinf(M))

    a = np.ones(n) / n
    b = np.ones(m) / m
    aa = a + 1e-1 * np.random.rand(n) / n
    bb = b + 1e-1 * np.random.rand(m) / m

    aa = aa / np.linalg.norm(aa, ord=1)
    bb = bb / np.linalg.norm(bb, ord=1)

    P = np.outer(aa, bb)
    P[row_idx, col_idx] = 0
    f = np.zeros(n)
    g = np.zeros(m)

    fsgw_val = []

    for p in range(fsgw_niter):

        D = np.zeros((n, m)) 
        non_zero_indices = np.argwhere(P != 0)
        for i, j in non_zero_indices:
            # Compute the contribution for each non-zero entry of P
            D += P[i, j] * (D1_norm[:, i, None] - D2_norm[None, j, :])**2
        fsgw = (1 - fsgw_alpha) * np.sum(M * P) + fsgw_alpha*(np.sum(D * P)
            +fsgw_gamma*(np.sum(a)+np.sum(b)-2*np.sum(P))+fsgw_eps*np.sum(P * (np.log(P+10**(-20)*np.ones((n,m)))-np.ones((n,m)))))
        fsgw_val.append(fsgw)

        D = 2*D

        D[row_idx, col_idx] = np.inf

        options = {
            'niter_sOT': 100,
            'f_init': np.zeros(n),
            'g_init': np.zeros(m),
            'penalty': 2,
            'stopthr': 1e-8
        }

        P, f, g = perform_sOT_log( (1 - fsgw_alpha)* M + fsgw_alpha*D, a, b, fsgw_eps, options)

    return P


def perform_sOT_log(G, a, b, eps, options):

    niter = options['niter_sOT']
    f     = options['f_init']
    g     = options['g_init']
    M     = options['penalty']
    stopthr = options['stopthr']

    err = 100
    q = 0
    while q <= niter and err > stopthr:
        fprev = f.copy()
        gprev = g.copy()
        f = np.minimum(eps * np.log(a) - eps * np.log(np.sum(np.exp((f[:, None] + g[None, :]  - G) / eps), axis=1) + np.finfo(float).eps) + f, M)
        g = np.minimum(eps * np.log(b) - eps * np.log(np.sum(np.exp((f[:, None] + g[None, :]  - G) / eps), axis=0) + np.finfo(float).eps) + g, M)
        # Check relative error
        if q % 10 == 0:
            err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
            err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
            err = 0.5 * (err_f + err_g)
        q = q + 1
        
    P = np.exp((f[:, None] + g[None, :] - G) / eps)
    
    return P, f, g


def vertex_with_most_edges(B):
    max_degree = max(dict(B.degree()).values())
    vertices = [v for v, d in B.degree() if d == max_degree]
    return vertices, max_degree

