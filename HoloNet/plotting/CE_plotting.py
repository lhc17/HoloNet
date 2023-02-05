from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import torch
from anndata._core.anndata import AnnData

from .base_plot import cell_type_level_network, feature_plot
from ..tools.CE_network_centrality import compute_ce_network_eigenvector_centrality, \
    compute_ce_network_degree_centrality


def ce_hotspot_plot(ce_tensor: torch.Tensor,
                    adata: AnnData,
                    lr_df: pd.DataFrame,
                    plot_lr: str,
                    scale: bool = True,
                    centrality_measure: str = 'degree',
                    consider_cell_role: str = 'sender_receiver',
                    max_iter: int = 100,
                    tol: float = 1e-4,
                    diff_thres: float = 0.05,
                    fname: Optional[Union[str, Path]] = None,
                    **kwargs,
                    ):
    """

    Plot the centrality of each spot in one LR CE network, representing the hotspot of one LR pair.

    Parameters
    ----------
    ce_tensor :
        A CE tensor (LR_pair_num * cell_num * cell_num)
    adata :
        Annotated data matrix.
    lr_df :
        The used preprocessed LR-gene dataframe, must contain the 'LR_pair' column.
    plot_lr :
        The LR pair (in the 'LR_pair' column of lr_df) need to be visualized.
    scale :
        If True, scale the centrality to 0-1 when plotting
    centrality_measure :
        Select to use degree or eigenvector centrality.
    consider_cell_role :
        One value selected in 'receiver', 'sender' and 'sender_receiver',
        determining the function calculating the in-degrees, out-degrees, or the sum of them,
        See in 'compute_ce_network_degree_centrality' function.
    max_iter :
        Maximum iteration number for get stable eigenvector centrality.
        See in 'compute_ce_network_eigenvector_centrality' function.
    tol :
        Defining stablity, we need the eigenvector centralities similar to the last iteration in how many cells.
        See in 'compute_ce_network_eigenvector_centrality' function.
    diff_thres :
        Defining stablity, the centrality of cells differs less than how much we consider similar.
        See in 'compute_ce_network_eigenvector_centrality' function.
    fname :
        The output file name. If None, not save the figure. Note that not add path name, sc.pl.spatial will add 'show'
        before the file name.
    kwargs :
        Other paramters in 'feature_plot' function.

    """

    if centrality_measure not in ['eigenvector', 'degree']:
        raise OSError('Please select centrality measure methods in \'eigenvector\' and \'degree\'')

    if centrality_measure == 'degree':
        cell_cci_centrality = compute_ce_network_degree_centrality(ce_tensor,
                                                                   consider_cell_role=consider_cell_role)
    else:
        cell_cci_centrality = compute_ce_network_eigenvector_centrality(ce_tensor,
                                                                        max_iter=max_iter,
                                                                        tol=tol,
                                                                        diff_thres=diff_thres)

    feature_plot(feature_mat=cell_cci_centrality,
                 adata=adata,
                 feature_names=lr_df['LR_Pair'],
                 plot_feature=plot_lr,
                 scale=scale,
                 fname=fname,
                 **kwargs)


def ce_cell_type_network_plot(ce_tensor: torch.Tensor,
                              cell_type_tensor: Union[torch.Tensor, np.ndarray, pd.DataFrame],
                              cell_type_names: List[str],
                              lr_df: pd.DataFrame,
                              plot_lr: str,
                              fname: Optional[Union[str, Path]] = None,
                              **kwargs,
                              ):
    """

    Plot the cell-type-level CE network of a certain LR pair.

    Parameters
    ----------
    ce_tensor :
        A CE tensor (LR_pair_num * cell_num * cell_num)
    cell_type_tensor :
        cell_num * cell_type_num, derived from 'get_continuous_cell_type_tensor' or
        'get_one_hot_cell_type_tensor' function.
    cell_type_names :
        The list of cell-type names.
    lr_df :
        The used preprocessed LR-gene dataframe, must contain the 'LR_pair' column.
    plot_lr :
        The LR pair (in the 'LR_pair' column of lr_df) need to be visualized.
    fname :
        The output file name. If None, not save the figure.
    kwargs :
        Other parameters in 'cell_type_level_network' function.

    Returns
    -------
    The cell-type-level CE network matrix.

    """

    cell_type_tensor = torch.FloatTensor(np.array(cell_type_tensor))
    if plot_lr == 'all':
        cell_type_impact = []
        for plot_LR_id in range(len(list(lr_df.LR_Pair))):
            cell_type_impact.append(ce_tensor[plot_LR_id].matmul(cell_type_tensor))
        cell_type_impact = torch.stack(cell_type_impact).mean(0)
    else:
        plot_LR_id = list(lr_df.LR_Pair).index(plot_lr)
        cell_type_impact = ce_tensor[plot_LR_id].matmul(cell_type_tensor)

    SR_network = cell_type_tensor.T.matmul(cell_type_impact).detach().numpy().T
    row, col = np.diag_indices_from(SR_network)
    SR_network[row, col] = 0
    SR_network = (SR_network - SR_network.min()) / (SR_network.max() - SR_network.min() + 1e-6)

    cell_type_level_network(sr_network=SR_network,
                            cell_type_names=cell_type_names,
                            fname=fname,
                            **kwargs)

    SR_network = pd.DataFrame(SR_network, index=cell_type_names, columns=cell_type_names)

    return SR_network
