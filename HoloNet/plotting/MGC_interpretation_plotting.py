from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm

from .base_plot import cell_type_level_network
from ..predicting.MGC_model import MGC_Model


def lr_rank_in_mgc(trained_MGC_model_list: List[MGC_Model],
                   lr_df: pd.DataFrame,
                   repeat_filter_num: int = 0,
                   repeat_attention_scale: bool = True,
                   plot_lr_num: int = 15,
                   fname: Optional[Union[str, Path]] = None,
                   display: bool = True,
                   plot_cluster: bool = True,
                   cluster_col: str = 'cluster',
                   ) -> pd.DataFrame:
    """

    Analyze the MGC attention value corresponding to each LR pair, and plot the LR pairs serving as the core mediators
    of FCEs.

    Parameters
    ----------
    trained_MGC_model_list :
        A list of trained MGC model for generating the expression of one target gene.
    lr_df :
        The used preprocessed LR-gene dataframe, must contain the 'LR_pair' column.
    repeat_filter_num :
        The number of repetitions to be filtered out. If the attention obtained from a certain training
        has too low variance, delete the training result. If 0, not filter any training.
    repeat_attention_scale :
        If True, scale the attention of each training to 0-1.
    plot_lr_num :
        The number of LR pair displayed in the output figure.
    fname :
        The output file name. If None, not save the figure.
    display :
        If False, not plot the figure.
    plot_cluster :
        If True, plot the cluster information of LR pairs as the color bar of heatmap.
    cluster_col :
        The columns in lr_df dataframe storing the clustering results of each LR pair,
        which is necessary if plot_cluster == True.

    Returns
    -------
    lr_df added column 'MGC_layer_attention', which contains the attention value of each LR pair.

    """
    repeat_num = len(trained_MGC_model_list)
    layer_attention_final = [trained_MGC_model_list[i].mgc.layer_attention.data for i in range(repeat_num)]
    layer_attention_torch = torch.stack(layer_attention_final)

    used_repeat = np.sort(np.argsort(abs(layer_attention_torch).var(1))[repeat_filter_num:])

    df1 = pd.DataFrame(np.array(abs(layer_attention_torch)).T[:, used_repeat],
                       index=lr_df.LR_Pair)
    if repeat_attention_scale:
        df1 = (df1 - df1.min(0)) / (df1.max(0) - df1.min(0))
    tmp = pd.DataFrame(df1.mean(1), columns=['weight_sum']).sort_values(by='weight_sum', ascending=False)
    tmp['LR_Pair'] = tmp.index

    related_LR_df_result = lr_df.copy()
    related_LR_df_result.index = lr_df.LR_Pair
    related_LR_df_result['MGC_layer_attention'] = pd.DataFrame(df1.mean(1), columns=['weight_sum']).weight_sum
    related_LR_df_result.index = lr_df.index
    related_LR_df_result = related_LR_df_result.sort_values(by='MGC_layer_attention', ascending=False)

    plt.figure(figsize=(4, 4))

    first_col = 0
    if plot_cluster:
        first_col = 1
        if cluster_col not in lr_df.columns:
            raise Exception('Not detect clustering results')

    grid = plt.GridSpec(1, 12 + first_col, wspace=0.1)

    if plot_cluster:
        axes0 = plt.subplot(grid[0, 0])
        cluster_df = pd.DataFrame(np.array(lr_df[cluster_col]), index=lr_df.LR_Pair)

        axes0 = sns.heatmap(data=cluster_df.loc[list(tmp.index)[:plot_lr_num]], annot=True,
                            cmap="Accent", cbar=False)

    axes1 = plt.subplot(grid[0, first_col:9 + first_col])

    tmp1 = df1.loc[list(tmp.index)[:plot_lr_num]]
    tmp1 = (tmp1 - tmp1.min().min()) / (tmp1.max().max() - tmp1.min().min())

    axes1 = sns.heatmap(data=tmp1, vmax=1.15, cmap="Oranges", cbar=False)
    if plot_cluster:
        axes1.set(yticklabels=[])
        axes1.set(ylabel=None)
    axes1.set(xlabel=None)

    axes2 = plt.subplot(grid[0, 9 + first_col:12 + first_col])

    axes2 = sns.barplot(x='weight_sum', y='LR_Pair', data=tmp.head(plot_lr_num))
    axes2.set(yticklabels=[])
    axes2.set(ylabel=None)
    axes2.set(xlabel=None)
    axes2.tick_params(left=False)

    if fname is not None:
        plt.savefig(fname)
    if display:
        plt.show()
    plt.close()

    return related_LR_df_result


def fce_cell_type_network_plot(trained_MGC_model_list: List[MGC_Model],
                               lr_df: pd.DataFrame,
                               X: torch.Tensor,
                               adj: torch.Tensor,
                               cell_type_names: List[str],
                               plot_lr: str,
                               **kwargs,
                               ) -> pd.DataFrame:
    """

    Display the cell-type-level FCE network of a certain LR pair (or all LR pairs) for a certain target gene.

    Parameters
    ----------
    trained_MGC_model_list :
        A list of trained MGC model for generating the expression of one target gene.
    lr_df :
        The used preprocessed LR-gene dataframe, must contain the 'LR_pair' column.
    X :
        The feature matrix used as the input of the trained_MGC_model_list.
    adj :
        The adjacency matrix used as the input of the trained_MGC_model_list.
    cell_type_names :
        The list of cell-type names.
    plot_lr :
        The LR pair (in the 'LR_pair' column of lr_df) need to be visualized.
    kwargs :
        Other parameters in 'cell_type_level_network' function.

    Returns
    -------
    The cell-type-level FCE network matrix.

    """
    SR_network_list = []
    cell_type_impact_list = []
    for i in tqdm(range(len(trained_MGC_model_list))):
        model = trained_MGC_model_list[i]
        if plot_lr == 'all':
            cell_type_impact = []
            for plot_LR_id in range(len(list(lr_df.LR_Pair))):
                cell_type_impact.append(adj[plot_LR_id].matmul(X).mul(model.mgc.weight[plot_LR_id].sum(1)) *
                                        abs(model.mgc.layer_attention[plot_LR_id]))
            cell_type_impact = torch.stack(cell_type_impact).mean(0)
        else:
            plot_LR_id = list(lr_df.LR_Pair).index(plot_lr)
            cell_type_impact = adj[plot_LR_id].matmul(X).mul(model.mgc.weight[plot_LR_id].sum(1))
        cell_type_impact = F.relu(cell_type_impact)

        SR_network = X.T.matmul(cell_type_impact).detach().numpy().T
        row, col = np.diag_indices_from(SR_network)
        SR_network[row, col] = 0
        SR_network_list.append((SR_network - SR_network.min()) / (SR_network.max() - SR_network.min() + 1e-6))
        cell_type_impact = cell_type_impact.detach().numpy()
        cell_type_impact_list.append((cell_type_impact - cell_type_impact.min()) / 
                                     (cell_type_impact.max() - cell_type_impact.min() + 1e-6))

    SR_network = np.stack(SR_network_list).mean(0)
    cell_type_impact = np.stack(cell_type_impact_list).mean(0)
    
    SR_network = (SR_network - SR_network.min()) / (SR_network.max() - SR_network.min() + 1e-6)

    cell_type_level_network(sr_network=SR_network,
                            cell_type_names=cell_type_names,
                            **kwargs)

    SR_network = pd.DataFrame(SR_network, index=cell_type_names, columns=cell_type_names)

    return SR_network, cell_type_impact


def delta_e_proportion(trained_MGC_model_list: List[MGC_Model],
                       X: torch.Tensor,
                       adj: torch.Tensor,
                       cell_type_names: List[str],
                       palette: Optional[dict] = None,
                       fname: Optional[Union[str, Path]] = None,
                       display: bool = True,
                       low_ylim: Optional[float] = None,
                       high_ylim: Optional[float] = None,
                       **kwargs,
                       ) -> pd.DataFrame:
    """

    Plotting the proportion of delta_e in the sum of delta_e and e_0 in each cell-type.

    Parameters
    ----------
    trained_MGC_model_list :
        A list of trained MGC model for generating the expression of one target gene.
    X :
        The feature matrix used as the input of the trained_MGC_model_list.
    adj :
        The adjacency matrix used as the input of the trained_MGC_model_list.
    cell_type_names :
        The list of cell-type names.
    palette :
        The color dict for each cell-type.
    fname :
        The output file name. If None, not save the figure.
    display :
        If False, not plot the figure.
    low_ylim :
        The low boundary of y axis.
    high_ylim :
        The high boundary of y axis.
    kwargs :
        Other parameters in seaborn.barplot

    Returns
    -------
    The dataframe containing the proportion of delta_e in the sum of delta_e and e_0 in each cell-type.

    """

    ce_list = []
    b_list = []
      
    for i in tqdm(range(len(trained_MGC_model_list))):
        model = trained_MGC_model_list[i]
        x = model.mgc(adj.matmul(X))
        x = F.relu(x)

        ce = x.matmul(model.linear_ce.weight.T).sum(1)
        b = X.matmul(model.linear_b.weight.T).sum(1)

        ce_list.append(ce)
        b_list.append(b)

    tmp_df = pd.DataFrame()
    b_result = abs(X.T.mul(torch.stack(b_list).mean(0)).sum(1).detach().numpy())
    ce_result = abs(X.T.mul(torch.stack(ce_list).mean(0)).sum(1).detach().numpy())

    tmp_df['delta_e_proportion'] = ce_result / (b_result + ce_result)
    tmp_df['cell_type'] = cell_type_names

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x='cell_type', y='delta_e_proportion', data=tmp_df, palette=palette, **kwargs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    proportion_range = tmp_df['delta_e_proportion'].max() - tmp_df['delta_e_proportion'].min()
    if low_ylim is None:
        low_ylim = max(round(tmp_df['delta_e_proportion'].min() * 20) / 20 - proportion_range/2, 0)
    if high_ylim is None:
        high_ylim = min(round(tmp_df['delta_e_proportion'].max() * 20) / 20 + proportion_range/2, 1)
    ax.set(ylim=(low_ylim, high_ylim))

    if fname is not None:
        plt.savefig(fname)
    if display:
        plt.show()
    plt.close()

    return tmp_df
