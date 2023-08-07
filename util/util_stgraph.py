
import numpy as np
import scipy.sparse as sp
import torch


def calculate_random_walk_matrix(adj_mx, args):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    i = [list(random_walk_mx.row), list(random_walk_mx.col)]
    v = random_walk_mx.data
    s = torch.sparse_coo_tensor(i, v, (adj_mx.shape[0], adj_mx.shape[1]))
    return s.to(args.device)


def adj2lap(A):
    A = A + torch.eye(A.shape[0])
    D_sqrt_inv = torch.diag(1 / torch.sqrt(A.sum(dim=1)))
    A = D_sqrt_inv @ A @ D_sqrt_inv
    return A
