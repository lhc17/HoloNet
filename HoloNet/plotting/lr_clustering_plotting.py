from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
import sklearn
from anndata._core.anndata import AnnData
from scipy.cluster.hierarchy import dendrogram

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

    if len(str(lr_df.index[0])) > 1:
        LR_cluster_data.index = lr_df.LR_Pair
        LR_cluster_data['cluster'] = cluster
    else:
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


def lr_clustering_dendrogram(model: sklearn.cluster._agglomerative.AgglomerativeClustering,
                             lr_df: pd.DataFrame,
                             plot_lr_list: Optional[List[str]] = None,
                             dflt_col: str = '#000000',
                             colors: Optional[List[str]] = None,
                             fname: Optional[Union[str, Path]] = None,
                             ):
    """

    Plot dendrogram plot for the hierarchical clustering model.

    Parameters
    ----------
    model :
        Trained hierarchical clustering model.
    lr_df :
        A preprocessed LR-gene dataframe.
    plot_lr_list :
        Lr pair on interest, which will be plot in dendrogram plot
    dflt_col :
        Color of root.
    colors :
        Colors of each cluster.
    fname :
        The output file name. If None, not save the figure.

    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    if colors is None:
        colors = ["#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#a65628", "#ffff33", "#999999", "#fdbf6f"]

    D_leaf_colors = {}
    for i, label in enumerate(model.labels_):
        D_leaf_colors['attr_' + str(i)] = colors[label]

    link_cols = {}
    for i, i12 in enumerate(linkage_matrix[:, :2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(linkage_matrix) else D_leaf_colors["attr_%d" % x]
                  for x in i12)
        link_cols[i + 1 + len(linkage_matrix)] = c1 if c1 == c2 else dflt_col

    tmp_lr_list = []
    if plot_lr_list is not None:
        for i in list(lr_df['LR_Pair']):
            if i in plot_lr_list:
                tmp_lr_list.append(i)
            else:
                tmp_lr_list.append('')

    fig, ax = plt.subplots(figsize=(6, 4))
    D = dendrogram(Z=linkage_matrix, labels=tmp_lr_list, color_threshold=None,
                   leaf_font_size=8, link_color_func=lambda x: link_cols[x], ax=ax)
    plt.xticks(rotation=45, ha='right')
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
                         plot_feature=cluster_i, **kwargs)

    else:
        for i, cluster_i in enumerate(clusters):
            feature_plot(torch.stack(tmp_list), adata, feature_names=clusters,
                         plot_feature=cluster_i,
                         fname=cluster_i + '_' + fname, **kwargs)
