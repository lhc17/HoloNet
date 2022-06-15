from copy import copy

import numpy as np
import torch


def compute_ce_network_eigenvector_centrality(ce_tensor: torch.Tensor,
                                              max_iter: int = 100,
                                              tol: float = 1e-4,
                                              diff_thres: float = 0.05,
                                              ) -> torch.Tensor:
    """
    Calculate the eigenvector centrality of each cell in the CE network.

    Parameters
    ----------
    ce_tensor :
        A CE tensor (LR_pair_num * cell_num * cell_num)
    max_iter :
        Maximum iteration number for get stable eigenvector centrality.
    tol :
        Defining stablity, we need the eigenvector centralities similar to the last iteration in how many cells.
    diff_thres :
        Defining stablity, the centrality of cells differs less than how much we consider similar.

    Returns
    -------
    A tensor (LR_num * cell_num) for the eigenvector centrality of each cell according to each LR pair.

    """
    undirected_ce_tensor = ce_tensor + ce_tensor.permute(0, 2, 1)
    cell_num = ce_tensor.shape[2]
    LR_num = ce_tensor.shape[0]
    x = torch.ones([cell_num, LR_num]) / cell_num

    for i in range(max_iter):
        xlast = copy(x)
        x = x.T.unsqueeze(1).matmul(undirected_ce_tensor.to(torch.float32)).squeeze(1).T
        norm = (x ** 2).sum(0)
        x = x / norm
        if len(np.where((abs(xlast - x).sum(0)) > diff_thres)[0]) < cell_num * tol:
            break
    return x.T


def compute_ce_network_degree_centrality(ce_tensor: torch.Tensor,
                                         consider_cell_role: str = 'sender_receiver',
                                         ) -> torch.Tensor:
    """
    Calculate the degree centrality of each cell in the CE network.

    Parameters
    ----------
    ce_tensor :
        A CE tensor (LR_pair_num * cell_num * cell_num)
    consider_cell_role :
        One value selected in 'receiver', 'sender' and 'sender_receiver',
        determining the function calculating the in-degrees, out-degrees, or the sum of them,

    Returns
    -------
    A tensor (LR_num * cell_num) for the degree centrality of each cell according to each LR pair.

    """
    if consider_cell_role not in ['sender_receiver', 'receiver', 'sender']:
        raise OSError('Please select consider_cell_role in \'sender_receiver\', \'receiver\', \'sender\'')

    if consider_cell_role == 'sender_receiver':
        cell_cci_centrality = ce_tensor.sum(2) + ce_tensor.sum(1)

    if consider_cell_role == 'receiver':
        cell_cci_centrality = ce_tensor.sum(2)

    if consider_cell_role == 'sender':
        cell_cci_centrality = ce_tensor.sum(1)

    return cell_cci_centrality
