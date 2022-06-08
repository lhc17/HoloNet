from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from anndata._core.anndata import AnnData
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import LabelBinarizer


def adj_normalize(adj: torch.Tensor,
                  cell_type_tensor: torch.Tensor,
                  only_between_cell_type: bool = True,
                  ) -> torch.Tensor:
    """

    Process the relationship among cells (such as multi-view CE network or spatial proximity matrix) to form the
    normalized adjancency matrix as the input of GNN.

    Parameters
    ----------
    cell_type_tensor :
        cell_num * cell_type_num, derived from 'get_continuous_cell_type_tensor' or
        'get_one_hot_cell_type_tensor' function.
    adj :
        view_num * cell_num * cell_num, can be multi-view CE network or spatial proximity matrix
    only_between_cell_type :
        if True, will remove the edges between two cells with the same cell-type, to control the confounding factors.

    Returns
    -------
    Normalized adjancency matrix as the input of GNN.

    """

    if not isinstance(adj, torch.FloatTensor):
        adj = torch.FloatTensor(adj)
    if len(adj.shape) == 2:
        adj = adj.unsqueeze(0)

    adj = adj + adj.permute(0, 2, 1)

    if only_between_cell_type:
        if not isinstance(cell_type_tensor, torch.FloatTensor):
            cell_type_tensor = torch.FloatTensor(cell_type_tensor)
        same_cell_type_mask = cell_type_tensor.matmul(cell_type_tensor.T)
        adj = (1 - same_cell_type_mask).mul(adj)

    D = (adj + 1e-6).sum(1).unsqueeze(2)
    adj = (D ** (-1 / 2)).mul(adj).permute(0, 2, 1).mul(D ** (-1 / 2))

    adj = torch.eye(adj.shape[1]) + adj
    adj = adj.to(torch.float32)

    return adj


def train_test_mask(cell_num: int,
                    train_set_ratio: float = 0.6,
                    val_set_ratio: float = 0.2,
                    ) -> Tuple[List[int], List[int], List[int]]:
    """

    Get the index of cells using as the training set, the validation set, or the testing set.
    Set the train_set_ratio and val_set_ratio, the last part is the testing set.

    Parameters
    ----------
    cell_num :
        The number of cell.
    train_set_ratio :
        A value from 0-1. The ratio of cells using as the training set.
    val_set_ratio :
        A value from 0-1. The ratio of cells using as the validation set.

    Returns
    -------
    Three list of cell index, respectively for the training set, the validation set, and the testing set.

    """

    train_cell_num = round(cell_num * train_set_ratio)
    val_cell_num = round(cell_num * val_set_ratio)

    train_mask = np.random.choice(list(range(cell_num)), train_cell_num + val_cell_num, replace=False)
    val_mask = list(np.random.choice(train_mask, val_cell_num, replace=False))

    test_mask = list(set(range(cell_num)) - set(train_mask))
    train_mask = list(set(train_mask) - set(val_mask))

    return train_mask, test_mask, val_mask


def get_continuous_cell_type_tensor(adata: AnnData,
                                    continuous_cell_type_slot: str = 'predicted_cell_type',
                                    not_used_col: List[str] = ['max'],
                                    ) -> Tuple[torch.Tensor, List[str]]:
    """

    Get continuous cell-type information matrix, used as the feature matrix of GNN.
    The cell-type information always is the proportion of cell-types in each spot, derived from deconvolution methods
    and stored at one slot in adata.obsm.

    Parameters
    ----------
    adata :
        Annotated data matrix with cell-type information.
    continuous_cell_type_slot :
        The slot name of continuous cell-type information in in adata.obsm.
    not_used_col :
        The columns in continuous cell-type information matrix not used

    Returns
    -------
        cell_type_tensor: cell_num * cell_type_num
        cell_type_names: used cell-type names

    """

    cell_type_df = adata.obsm[continuous_cell_type_slot].drop(not_used_col, axis=1)
    cell_type_tensor = torch.FloatTensor(np.array(cell_type_df))
    cell_type_names = list(cell_type_df.columns)

    return cell_type_tensor, cell_type_names


def get_one_hot_cell_type_tensor(adata: AnnData,
                                 categorical_cell_type_col: str = 'predictions',
                                 ) -> Tuple[torch.Tensor, List[str]]:
    """

    Get categorical cell-type labels and one-hot encoded into a matrix, used as the feature matrix of GNN.
    The categorical cell-type labels are stored at one slot in adata.obs.

    Parameters
    ----------
    adata :
        Annotated data matrix with cell-type information.
    categorical_cell_type_col :
        The column name of categorical cell-type labels in in adata.obs.

    Returns
    -------
        cell_type_tensor: cell_num * cell_type_num
        cell_type_names: used cell-type names

    """
    cell_type = list(adata.obs[categorical_cell_type_col])
    nan_index = [i for i, n in enumerate(cell_type) if not isinstance(n, str)]
    cell_display = np.zeros(adata.shape[0])
    cell_display[nan_index] = 1

    adjacency_mat = find_neighbors(adata)
    for i in range(len(nan_index)):
        a = np.where(adjacency_mat[nan_index[i], :] != 0)[0]
        cell_type[nan_index[i]] = pd.value_counts(adata.obs[categorical_cell_type_col][a]).index[0]

    one_hot = LabelBinarizer()
    one_hot_cell_type = one_hot.fit_transform(cell_type)

    cell_type_index = list(one_hot.classes_)
    # cell_type_index = np.array([re.sub("_", " ",i).title() for i in cell_type_index])

    return torch.FloatTensor(one_hot_cell_type), cell_type_index


def find_neighbors(adata):
    position_mat = adata.obsm['spatial']
    dist_mat = squareform(pdist(position_mat, metric='euclidean'))

    boundry = 1.4 * dist_mat[dist_mat > 0].min()
    dist_mat[dist_mat < boundry] = 1
    dist_mat[dist_mat >= boundry] = 0
    dist_mat[range(dist_mat.shape[0]), range(dist_mat.shape[0])] = 0
    return dist_mat
