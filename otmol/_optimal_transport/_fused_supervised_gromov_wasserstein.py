import itertools
import numpy as np
from scipy import sparse
import networkx as nx

def perform_sOT_log(G, a, b, eps, options):

    niter = options['niter_sOT']
    f     = options['f_init']
    g     = options['g_init']
    M     = options['penalty']

    # Err = np.array([[1, 1]])

    for q in range(niter):   
        f = np.minimum(eps * np.log(a) - eps * np.log(np.sum(np.exp((f[:, None] + g[None, :]  - G) / eps), axis=1)+ 10**-20) + f, M)
        g = np.minimum(eps * np.log(b) - eps * np.log(np.sum(np.exp((f[:, None] + g[None, :]  - G) / eps), axis=0)+ 10**-20) + g, M)

    P = np.exp((f[:, None] + g[None, :] - G) / eps)
    
    return P, f, g

def vertex_with_most_edges(B):
    max_degree = max(dict(B.degree()).values())
    vertices = [v for v, d in B.degree() if d == max_degree]
    return vertices, max_degree

def fused_supervised_gromov_wasserstein(
    D1,
    D2,
    M,
    gw_cutoff = np.inf,
    w_cutoff = np.inf,
    fsgw_niter = 10,
    fsgw_eps = 0.01,
    fsgw_alpha = 0.1,
    fsgw_gamma = 2,
    
):
    n = D1.shape[0]
    m = D2.shape[0]
    P_idx = np.array( [[i, j] for i, j in itertools.product(range(n), range(m))], dtype=int )
    
    I = []
    J = []
    for i in range(len(P_idx)):
        D_tmp = (D1[P_idx[i][0],:,np.newaxis] - D2[P_idx[i][1],:])**2
        tmp_idx = np.where(D_tmp.flatten() <= gw_cutoff**2)[0]
        J.extend(list(tmp_idx))
        I.extend([i for _ in range(len(tmp_idx))])
    I = np.array(I, int)
    J = np.array(J, int)
    D = np.ones_like(I)
    A = sparse.coo_matrix((D, (I, J)), shape=(n*m, n*m))
    
    
    G = nx.from_scipy_sparse_array(A)
    tmp_idx = np.where(M.flatten() >=  w_cutoff)[0]
    G.remove_nodes_from(tmp_idx)

    
    M_flatten = M.flatten()
    zero_indices = set()
    G_copy = nx.complement(G)
    del G
    # with tqdm(total=G_copy.number_of_edges(), desc=f"Finding min vertex covering for cutoff_GW {gw_cutoff} and cutoff_CC {w_cutoff}") as pbar:
    while G_copy.edges:
        initial_edges = G_copy.number_of_edges()
        max_degree_vertices, max_degree = vertex_with_most_edges(G_copy)
        max_Cij_value = -float('inf')
        vertex_with_max_Cij = None
        for vertex in max_degree_vertices:
            if M_flatten[vertex] > max_Cij_value:
                max_Cij_value = M_flatten[vertex]
                vertex_with_max_Cij = vertex

        G_copy.remove_node(vertex_with_max_Cij)
        zero_indices.add(vertex_with_max_Cij)
        removed_edges = initial_edges - G_copy.number_of_edges()
            # pbar.update(removed_edges)
    del G_copy
    zero_indices = list(zero_indices) + list(tmp_idx)
    zero_indices = np.array(zero_indices)
    print("# of potential non-zeros in P:", n*m - len(zero_indices))
    
    row_idx = P_idx[zero_indices,0]
    col_idx = P_idx[zero_indices,1]

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
            D += P[i, j] * (D1[:, i, None] - D2[None, j, :])**2
        fsgw = (1 - fsgw_alpha) * np.sum(M * P) + fsgw_alpha*(np.sum(D * P)
            +fsgw_gamma*(np.sum(a)+np.sum(b)-2*np.sum(P))+fsgw_eps*np.sum(P * (np.log(P+10**(-20)*np.ones((n,m)))-np.ones((n,m)))))
        fsgw_val.append(fsgw)

        D = 2*D

        D[row_idx, col_idx] = np.inf

        options = {
            'niter_sOT': 10**4,
            'f_init': np.zeros(n),
            'g_init': np.zeros(m),
            'penalty': 2
        }

        P, f, g = perform_sOT_log( (1 - fsgw_alpha)* M + fsgw_alpha*D, a, b, fsgw_eps, options)

    return P