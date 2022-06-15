from typing import List

import numpy as np
import pandas as pd
import torch
from anndata._core.anndata import AnnData
from scipy.spatial.distance import pdist, squareform


def compute_ce_tensor(adata: AnnData,
                      lr_df: pd.DataFrame,
                      w_best: float,
                      distinguish: bool = True,
                      ) -> torch.Tensor:
    """\
    Calculate CE matrix for measuring the strength of communication between any pairs of cells, according to
    the edge weighting function.

    Parameters
    ----------
    adata
        Annotated data matrix.
    lr_df
        A preprocessed LR-gene dataframe.
        must contain three columns: 'Ligand_gene_symbol', 'Receptor_gene_symbol' and 'LR_pair'.
    w_best
        A distance parameter in edge weighting function controlling the covering region of ligands.
        'default_w_visium' function provides a recommended value of w_best.
    distinguish:
        If True, set the different w_best for secreted ligands and plasma-membrane-binding ligands.
    
    Returns
    -------
    A CE tensor (LR_pair_num * cell_num * cell_num)
    
    """

    dist_factor_tensor = distinguish_dist_factor_calculate(adata=adata,
                                                           lr_df=lr_df,
                                                           w_best=w_best,
                                                           distinguish=distinguish)

    expressed_ligand = lr_df.loc[:, 'Ligand_gene_symbol'].tolist()
    expressed_receptor = lr_df.loc[:, 'Receptor_gene_symbol'].tolist()

    expressed_ligand_tensor = get_gene_expr_tensor(adata, expressed_ligand)
    expressed_receptor_tensor = get_gene_expr_tensor(adata, expressed_receptor).permute(0, 2, 1)

    ce_tensor = expressed_ligand_tensor.mul(dist_factor_tensor).mul(expressed_receptor_tensor)
    ce_tensor = ce_tensor / ce_tensor.mean((1, 2)).unsqueeze(1).unsqueeze(2)

    return ce_tensor.to(torch.float32)


def dist_factor_calculate(adata: AnnData,
                          w_best: float,
                          obsm_spatial_slot: str = 'spatial',
                          ) -> torch.Tensor:
    """

    Parameters
    ----------
    adata :
        Annotated data matrix.
    w_best :
        A distance parameter in edge weighting function controlling the covering region of ligands.
        'default_w_visium' function provides a recommended value of w_best.
    obsm_spatial_slot :
        The slot name storing the spatial position information of each spot。

    Returns
    -------
    A tensor describing the distance factor.

    """
    position_mat = adata.obsm[obsm_spatial_slot]
    dist_mat = squareform(pdist(position_mat, metric='euclidean'))
    dist_factor = dist_mat / w_best

    dist_factor = np.exp((-1) * dist_factor * dist_factor)
    dist_factor = torch.tensor(dist_factor).to(torch.float32)

    return dist_factor


def distinguish_dist_factor_calculate(adata: AnnData,
                                      lr_df: pd.DataFrame,
                                      w_best: float,
                                      distinguish=False,
                                      ) -> torch.Tensor:
    if distinguish:
        w_best2 = w_best * 2
        dist_factor1 = dist_factor_calculate(adata=adata, w_best=w_best, )
        dist_factor2 = dist_factor_calculate(adata=adata, w_best=w_best2, )
        dist_factor_tensor = dist_factor1.repeat(lr_df.shape[0], 1, 1)
        secreted_index = lr_df[lr_df.Ligand_location == 'secreted'].index
        dist_factor_tensor[secreted_index, :, :] = dist_factor2
    else:
        dist_factor_tensor = dist_factor_calculate(adata, w_best=w_best, )

    return dist_factor_tensor


def get_gene_expr_tensor(adata: AnnData,
                         gene_name: List[str],
                         ) -> torch.Tensor:
    gene_expr_mat = adata[:, gene_name].X.toarray().astype(np.float32)
    gene_expr_tensor = torch.tensor(np.expand_dims(gene_expr_mat, 2)).permute(1, 2, 0)

    return gene_expr_tensor
