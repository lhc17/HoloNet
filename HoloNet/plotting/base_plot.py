from pathlib import Path
from typing import Union, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata._core.anndata import AnnData
from matplotlib.colors import Colormap


def feature_plot(feature_mat: Union[torch.Tensor, np.ndarray],
                 adata: AnnData,
                 feature_names: List[str],
                 plot_feature: str,
                 size: float = 1.4,
                 fname: Optional[str] = None,
                 cutoff: Optional[str] = None,
                 scale: bool = False,
                 cell_alpha: float = 0.7,
                 cmap: Union[Colormap, str] = 'Spectral_r',
                 **kwargs,
                 ) -> AnnData:
    """

    Plot one feature of each cell at the spatial position.

    Parameters
    ----------
    feature_mat :
        A matrix containing features in each cell (CE centrality, or generated gene expression et al).
    adata :
        Annotated data matrix.
    feature_names :
        The names of all features.
    plot_feature :
        The name of feature for plotting.
    size :
        The size of each spot, see sc.pl.spatial in Scanpy.
    fname :
        The output file name. If None, not save the figure. Note that not add path name, sc.pl.spatial will add 'show'
        before the file name.
    cutoff :
        The cutoff value (percentage) of the plotted feature.
    scale :
        If True, will scale the feature value to make the maximum into 1.
    cell_alpha :
        The alpha value of each spot.
    cmap :
        The color map of feature value of each spot, defaultly as 'Spectral_r'.
    kwargs :
        Other parameters in sc.pl.spatial

    """

    if len(feature_mat.shape) == 1:
        feature_mat = feature_mat.unsqueeze(1)

    assert len(np.where(np.array(feature_mat.shape) == adata.shape[0])[0]) >= 1

    if int(np.where(np.array(feature_mat.shape) == adata.shape[0])[0]) == 1:
        feature_mat = feature_mat.T

    var_mat = pd.DataFrame(feature_names)
    var_mat.columns = ['feature_names']
    var_mat.index = var_mat.feature_names

    if scale:
        feature_mat = (feature_mat - feature_mat.min()) / (feature_mat.max() - feature_mat.min())

    plot_feature_adata = AnnData(np.array(feature_mat), obs=adata.obs,
                                      var=var_mat, uns=adata.uns, obsm=adata.obsm, dtype='float64')

    if cutoff is not None:
        max_value = cutoff * plot_feature_adata[:, plot_feature].X.max()
        plot_feature_adata.X[plot_feature_adata.X > max_value] = max_value

    sc.pl.spatial(plot_feature_adata, color=plot_feature, alpha=cell_alpha,
                  size=size, cmap=cmap, save=fname, **kwargs)

    return plot_feature_adata


def cell_type_level_network(sr_network: np.ndarray,
                            cell_type_names: List[str],
                            edge_thres: float = 0.05,
                            edge_width_times: float = 1.5,
                            linewidths: float = 1.0,
                            edgecolors: str = 'black',
                            connectionstyle: str = 'arc3, rad = 0.15',
                            arrowstyle: str = '-|>',
                            palette: Optional[dict] = None,
                            fname: Optional[Union[str, Path]] = None,
                            display: bool = True,
                            **kwargs,
                            ):
    """

    Plot the relationship network (CEs or FCEs) between cell-types.

    Parameters
    ----------
    sr_network :
        The sender-receiver network among cell-types (cell_type_num * cell_type_num). The mox edge weight need to be 1.
    cell_type_names :
        The cell-type name list
    edge_thres :
        The plot threshold of edges. The edge with weight lower than th threshold will not be display in the plot.
    edge_width_times :
        Increase the weight of the edge by a certain factor as the thickness of the arrow。
    linewidths :
        Widths of node borders
    edgecolors :
        Colors of node borders
    connectionstyle :
        Pass the connectionstyle parameter to create curved arc of rounding radius rad.
        See in matplotlib.patches.ConnectionStyle and matplotlib.patches.FancyArrowPatch
    arrowstyle :
        For directed graphs, choose the style of the arrowsheads. See in matplotlib.patches.ArrowStyle
    palette :
        The color dict for each cell-type.
    fname :
        The output file name. If None, not save the figure.
    display :
        If False, will not display the plot.
    kwargs :
        Other parameters in networkx.draw.

    """

    sr_network_graph = nx.DiGraph()
    sr_network_graph.add_nodes_from(cell_type_names)
    plt.figure(figsize=(3, 3))
    for i in range(sr_network.shape[0]):
        for j in range(sr_network.shape[0]):
            if sr_network[i][j] > edge_thres:
                sr_network_graph.add_weighted_edges_from(
                    [(cell_type_names[i], cell_type_names[j], sr_network[i][j])])

    if palette is not None:
        node_color = [palette[i] for i in cell_type_names]
    else:
        node_color = None

    nx.draw(sr_network_graph, with_labels=True, pos=nx.circular_layout(sr_network_graph),
            linewidths=linewidths, edgecolors=edgecolors,
            node_color=node_color,
            connectionstyle=connectionstyle, arrowstyle=arrowstyle,
            width=[float(v['weight'] * edge_width_times) for (r, c, v) in sr_network_graph.edges(data=True)], **kwargs)

    if fname is not None:
        plt.savefig(fname)
    if display:
        plt.show()
    plt.close()


def plot_cell_type_proportion(adata: AnnData,
                              plot_cell_type: str,
                              continuous_cell_type_slot: str = 'predicted_cell_type',
                              fname: Optional[Union[str, Path]] = None,
                              **kwargs,
                              ):
    """

    Plot the proportion of one cell-type in each spot.

    Parameters
    ----------
    adata :
        Annotated data matrix.
    plot_cell_type :
        The name of cell type for plotting.
    continuous_cell_type_slot :
        The slot name of continuous cell-type information in in adata.obsm.
    fname :
        The output file name. If None, not save the figure. Note that not add path name, sc.pl.spatial will add 'show'
        before the file name.
    kwargs :
        Other paremeters in 'feature_plot' function.

    """
    tmp = adata.obsm[continuous_cell_type_slot]
    feature_plot(np.array(tmp).T, adata, feature_names=list(tmp.columns),
                 plot_feature=plot_cell_type, fname=fname, **kwargs)
