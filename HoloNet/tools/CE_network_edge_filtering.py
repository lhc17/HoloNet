import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from anndata._core.anndata import AnnData
from tqdm import tqdm

from .CE_network_edge_weighting import distinguish_dist_factor_calculate
from .CE_network_edge_weighting import get_gene_expr_tensor, compute_ce_tensor


def filter_ce_tensor_connectomeDB(ce_tensor: torch.Tensor,
                                  adata: AnnData,
                                  lr_df: pd.DataFrame,
                                  w_best: float,
                                  n_pairs: int = 200,
                                  thres: float = 0.05,
                                  distinguish: bool = True,
                                  copy: bool = True,
                                  ) -> torch.Tensor:
    """
    Filter the edge in calculated CE tensor, removing the edges with low specificities.

    For each LR pair, select faked ligand and receptor genes, which have similar expression levels
    with the ligand and receptor gene in the dataset. Then calculate the background CE tensor using faked LR genes,

    Using permutation tests, require filtered edges with communication event strength larger than
    a proportion of background strengthes.

    Parameters
    ----------
    ce_tensor :
        Calculated CE tensor (LR_pair_num * cell_num * cell_num) by "compute_ce_tensor" function
    adata :
        Annotated data matrix.
    lr_df :
        A preprocessed LR-gene dataframe.
    w_best :
        A distance parameter in edge weighting function controlling the covering region of ligands.
        'default_w_visium' function provides a recommended value of w_best.
    n_pairs :
        The number of faked ligand and receptor genes.
    thres :
        We require filtered edges with communicatin event strength larger than a proportion of background strengthes.
        The parameter is the proportion.
    distinguish :
        If True, set the different w_best for secreted ligands and plasma-membrane-binding ligands.
    copy :
        If False, change the input ce_tensor and save memory consumption

    Returns
    -------
    A CE tensor which removed the edges with low specificities (LR_pair_num * cell_num * cell_num)

    """

    if copy:
        if ce_tensor.is_sparse:
            ce_tensor = deepcopy(ce_tensor.to_dense())
        else:
            ce_tensor = deepcopy(ce_tensor)
    all_genes = [item for item in adata.var_names.tolist()
                 if not (item.startswith("MT-") or item.startswith("MT_"))]
    means = adata.to_df()[all_genes].mean().sort_values()

    for i in tqdm(range(lr_df.shape[0])):
        dist_factor_tensor = distinguish_dist_factor_calculate(adata=adata,
                                                               lr_df=lr_df.iloc[i:i + 1, :].reset_index(drop=True),
                                                               w_best=w_best,
                                                               distinguish=distinguish)

        lr1 = lr_df.Ligand_gene_symbol[i]
        lr2 = lr_df.Receptor_gene_symbol[i]
        i1, i2 = means.index.get_loc(lr1), means.index.get_loc(lr2)
        i1, i2 = np.sort([i1, i2])
        im = np.argmin(abs(means.values - means.iloc[i1:i2].median()))

        selected = (
            abs(means - means.iloc[im])
                .sort_values()
                .drop([lr1, lr2])[: n_pairs * 2]
                .index.tolist()
        )
        faked_ligand = selected[-n_pairs:]
        faked_receptor = selected[:n_pairs]

        faked_expressed_ligand_tensor = get_gene_expr_tensor(adata, faked_ligand)
        faked_expressed_receptor_tensor = get_gene_expr_tensor(adata, faked_receptor).permute(0, 2, 1)
        faked_ce_tensor = faked_expressed_ligand_tensor.mul(dist_factor_tensor).mul(faked_expressed_receptor_tensor)

        expressed_ligand_tensor = get_gene_expr_tensor(adata, lr1)
        expressed_receptor_tensor = get_gene_expr_tensor(adata, lr2).permute(0, 2, 1)
        true_ce_tensor = expressed_ligand_tensor.mul(dist_factor_tensor).mul(expressed_receptor_tensor)

        tmp = (true_ce_tensor > faked_ce_tensor).sum(0) > n_pairs * (1 - thres)
        ce_tensor[i, :, :] = ce_tensor[i, :, :].mul(tmp)

    return ce_tensor


def filter_ce_tensor(ce_tensor: torch.Tensor,
                     adata: AnnData,
                     lr_df: pd.DataFrame,
                     elements_expr_df_dict: dict,
                     w_best: float,
                     n_pairs: int = 200,
                     thres: float = 0.1,
                     distinguish: bool = True,
                     copy: bool = True,
                     ) -> torch.Tensor:

    """

    Filter the edge in calculated CE tensor, removing the edges with low specificities.

    For each LR pair, select faked ligand and receptor genes, which have similar expression levels
    with the ligand and receptor gene in the dataset. Then calculate the background CE tensor using faked LR genes,

    Using permutation tests, require filtered edges with communication event strength larger than
    a proportion of background strengthes.

    Parameters
    ----------
    ce_tensor :
        Calculated CE tensor (LR_pair_num * cell_num * cell_num) by "compute_ce_tensor" function
    adata :
        Annotated data matrix.
    lr_df :
        A preprocessed LR-gene dataframe.
    elements_expr_df_dict :
        metadata from 'elements_expr_df_calculate' function.
    w_best :
        A distance parameter in edge weighting function controlling the covering region of ligands.
        'default_w_visium' function provides a recommended value of w_best.
    n_pairs :
        The number of faked ligand and receptor genes.
    thres :
        We require filtered edges with communicatin event strength larger than a proportion of background strengthes.
        The parameter is the proportion.
    distinguish :
        If True, set the different w_best for secreted ligands and plasma-membrane-binding ligands.
    copy :
        If False, change the input ce_tensor and save memory consumption.

    Returns
    -------
    A CE tensor which removed the edges with low specificities (LR_pair_num * cell_num * cell_num).

    """

    if copy:
        if ce_tensor.is_sparse:
            ce_tensor = deepcopy(ce_tensor.to_dense())
        else:
            ce_tensor = deepcopy(ce_tensor)

    all_genes = [item for item in adata.var_names.tolist()
                 if not (item.startswith("MT-") or item.startswith("MT_"))]

    means = adata.to_df()[all_genes].mean().sort_values()
    dist_factor_secreted = distinguish_dist_factor_calculate(adata=adata, lr_df=lr_df,
                                                             w_best=w_best * 2, distinguish=False)

    for i in tqdm(range(lr_df.shape[0])):
        faked_elements_expr_df_dict = \
            {'co_and_receptor_expr': get_faked_genes_expr(elements_expr_df_dict['co_and_receptor_expr'],
                                                          means, adata, n_pairs, lr_index=i),
             'receptor_expr': get_faked_genes_expr(elements_expr_df_dict['receptor_expr'],
                                                   means, adata, n_pairs, lr_index=i),
             'ligand_expr': get_faked_genes_expr(elements_expr_df_dict['ligand_expr'],
                                                 means, adata, n_pairs, lr_index=i),
             'AG_expr': get_faked_genes_expr(elements_expr_df_dict['AG_expr'], means, adata, n_pairs, lr_index=i),
             'AN_expr': get_faked_genes_expr(elements_expr_df_dict['AN_expr'], means, adata, n_pairs, lr_index=i), }

        faked_ce_tensor = compute_ce_tensor(lr_df.iloc[i:i + 1, :], w_best, faked_elements_expr_df_dict, adata,
                                            distinguish=distinguish, dist_factor_secreted=dist_factor_secreted)

        tmp = (ce_tensor[i, :, :] > faked_ce_tensor).sum(0) > n_pairs * (1 - thres)
        ce_tensor[i, :, :] = ce_tensor[i, :, :].mul(tmp)

    ce_tensor = ce_tensor / (ce_tensor.mean((1, 2)) + 1e-20).unsqueeze(1).unsqueeze(2)

    return torch.tensor(ce_tensor, dtype=torch.float32)


def get_faked_genes_expr(expr_df, means, adata, n_pairs, lr_index):

    selected = (
        abs(means - expr_df.iloc[:, lr_index].mean())
            .sort_values()[1: n_pairs]
            .index.tolist())

    random.seed(10)
    random.shuffle(selected)
    return adata[:, selected].to_df()
