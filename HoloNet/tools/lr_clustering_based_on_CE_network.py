from typing import Optional

import pandas as pd
import torch
from anndata._core.anndata import AnnData
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from .CE_network_centrality import compute_ce_network_eigenvector_centrality, compute_ce_network_degree_centrality
from .CE_network_edge_weighting import dist_factor_calculate
from ..predicting.MGC_training import seed_torch


def cluster_lr_based_on_ce(ce_tensor: torch.Tensor,
                           adata: AnnData,
                           lr_df: pd.DataFrame,
                           w_best: float,
                           n_clusters: int = 4,
                           cluster_based: str = 'node_centrality_euclidean',
                           centrality_measure: str = 'Eigenvector',
                           cell_cci_centrality: Optional[torch.tensor] = None,
                           **kwargs,
                           ) -> pd.DataFrame:
    """
    Cluster the LR pairs using the CE network.

    Parameters
    ----------
    ce_tensor :
        A CE tensor (LR_pair_num * cell_num * cell_num)
    adata :
        Annotated data matrix.
    lr_df :
        A preprocessed LR-gene dataframe.
        must contain three columns: 'Ligand_gene_symbol', 'Receptor_gene_symbol' and 'LR_pair'.
    w_best :
        A distance parameter in edge weighting function controlling the covering region of ligands.
        'default_w_visium' function provides a recommended value of w_best.
    n_clusters :
        Number of clusters
    cluster_based :
        Cluster methods, selected in 'node_centrality_euclidean', 'node_centrality_physical', and 'edge_overlap'
    centrality_measure :
        Compute methods
    cell_cci_centrality :
        Provided centrality tensor can save time.
        A tensor (LR_num * cell_num) for the  centrality of each cell according to each LR pair.
    kwargs :
        Parameters for compute centrality, see in 'compute_ce_network_degree_centrality' and
         'compute_ce_network_eigenvector_centrality' function

    Returns
    -------
    A LR-gene dataframe added the 'cluster' column.

    """
    
    seed_torch()
    
    if (cell_cci_centrality is None) and (cluster_based != 'edge_overlap'):
        if centrality_measure == 'Degree':
            cell_cci_centrality = compute_ce_network_degree_centrality(ce_tensor, **kwargs)
        if centrality_measure == 'Eigenvector':
            cell_cci_centrality = compute_ce_network_eigenvector_centrality(ce_tensor, **kwargs)

    if cluster_based == 'node_centrality_euclidean':
        Agglo = AgglomerativeClustering(n_clusters=n_clusters)
        Agglo = Agglo.fit(cell_cci_centrality)

    if cluster_based == 'node_centrality_physical':
        dist_factor = dist_factor_calculate(adata, w_best=w_best)
        physical_dist = cell_cci_centrality.matmul(dist_factor).matmul(cell_cci_centrality.T)
        physical_dist = (physical_dist / cell_cci_centrality.sum(1)).T / cell_cci_centrality.sum(1)

        Agglo = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
        Agglo = Agglo.fit(1 / physical_dist)

    if cluster_based == 'edge_overlap':

        dissimilarity = torch.zeros((ce_tensor.shape[0], ce_tensor.shape[0]))
        for i in tqdm(range(ce_tensor.shape[0])):
            tmp = abs(ce_tensor[i] - ce_tensor[i + 1:]) / (ce_tensor[i] + ce_tensor[i + 1:] + 1e-6)
            dissimilarity[i, i + 1:] = tmp.sum(-1).sum(-1)

        Agglo = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage='complete')
        dissimilarity = dissimilarity + dissimilarity.T
        Agglo = Agglo.fit(dissimilarity / dissimilarity.max())

    lr_df['cluster'] = Agglo.labels_

    return lr_df
