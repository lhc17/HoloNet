from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from anndata._core.anndata import AnnData

from .base_plot import feature_plot
from ..colorSchemes import color_sheet


def lr_umap(lr_df: pd.DataFrame,
            cell_cci_centrality: torch.Tensor,
            plot_lr_list: Optional[List[str]] = None,
            cluster_col: str = 'cluster',
            fname: Optional[Union[str, Path]] = None,
            **kwargs,
            ):
    """

    Display the LR pair in low-dimention space.

    Parameters
    ----------
    lr_df :
        A preprocessed LR-gene dataframe, must contain the columns 'LR_pair' and clustering results.
    cell_cci_centrality :
        A tensor (LR_num * cell_num) for the  centrality of each cell according to each LR pair.
    plot_lr_list :
        The LR pair list (in the 'LR_pair' column of lr_df) need to be visualized in the UMAP plot.
    cluster_col :
        The columns in lr_df dataframe storing the clustering results of each LR pair.
    fname :
        The output file name. If None, not save the figure.
    kwargs :
        Other parameters in plt.scatter

    """
    embedding = umap.UMAP(random_state=1).fit_transform(np.array(cell_cci_centrality))
    embedding = (embedding - embedding.min(0)) / (embedding.max(0) - embedding.min(0))

    cluster = lr_df[cluster_col]
    cluster_num = len(np.unique(cluster))

    LR_cluster_data = pd.DataFrame(embedding)
    LR_cluster_data.columns = ['dim1', 'dim2']
    LR_cluster_data['cluster'] = cluster
    LR_cluster_data.index = lr_df.LR_Pair

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    color_list = list(color_sheet.values())[:cluster_num]
    cluster_type = [i for i in range(cluster_num)]

    # plot
    for (cluster, color) in zip(cluster_type, color_list):
        label_name = "group " + str(cluster)
        ax.scatter(LR_cluster_data[LR_cluster_data['cluster'] == cluster]['dim1'],
                   LR_cluster_data[LR_cluster_data['cluster'] == cluster]['dim2'],
                   label=label_name,
                   c=color,
                   marker='o',
                   edgecolors='black',
                   **kwargs)

    ax.set_xlabel(r'UMAP1')
    ax.set_ylabel(r'UMAP2')
    plt.legend(ncol=1, labelspacing=0.05, bbox_to_anchor=(0.98, 0.22), frameon=False, handlelength=0.6)

    if plot_lr_list is not None:
        for lr_pair in plot_lr_list:
            plt.annotate(r'{}'.format(lr_pair),
                         xy=(LR_cluster_data.loc[lr_pair, 'dim1'],
                             LR_cluster_data.loc[lr_pair, 'dim2']), xycoords='data',
                         xytext=(-40, +50), textcoords='offset points', fontsize=8,
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"))

    # plt.gca().set_aspect('equal', 'datalim')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title('UMAP for clustered LR pairs')

    if fname is not None:
        plt.savefig(fname)
    plt.show()


def lr_cluster_ce_hotspot_plot(lr_df: pd.DataFrame,
                               cell_cci_centrality: torch.Tensor,
                               adata: AnnData,
                               cluster_col: str = 'cluster',
                               fname: Optional[Union[str, Path]] = None,
                               **kwargs,
                               ):
    """
    
    Plot the general CE hotspot of each ligand-receptor group.
    Clustering results need to be provided in 'cluster_col' of lr_df.
    
    Parameters
    ----------
    lr_df :
        A preprocessed LR-gene dataframe, must contain the columns 'LR_pair' and clustering results.
    cell_cci_centrality :
        A tensor (LR_num * cell_num) for the  centrality of each cell according to each LR pair.
    adata :
        Annotated data matrix.
    cluster_col :
        The columns in lr_df dataframe storing the clustering results of each LR pair.
    fname :
        The output file name. If None, not save the figure.
    kwargs :
        Other paramters in 'feature_plot' function.
        

    """
    tmp_list = []
    for i in np.unique(lr_df[cluster_col]):
        tmp_index = np.where(lr_df[cluster_col] == i)[0]
        tmp_list.append(cell_cci_centrality[tmp_index].sum(0))

    clusters = ['cluster' + str(i) for i in np.unique(lr_df[cluster_col])]
    
    if fname is None:
        for i, cluster_i in enumerate(clusters):
            feature_plot(torch.stack(tmp_list), adata, feature_names=clusters,
                         plot_feature=cluster_i,  **kwargs)
            
    else:
        for i, cluster_i in enumerate(clusters):
            feature_plot(torch.stack(tmp_list), adata, feature_names=clusters,
                         plot_feature=cluster_i,
                         fname=cluster_i + '_' + fname, **kwargs)
