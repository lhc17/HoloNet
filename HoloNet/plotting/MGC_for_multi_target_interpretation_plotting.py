import os
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from .MGC_interpretation_plotting import lr_rank_in_mgc, fce_cell_type_network_plot
from ..predicting.MGC_model import MGC_Model


def detect_pathway_related_genes(trained_MGC_model_list: List[List[MGC_Model]],
                                 lr_df: pd.DataFrame,
                                 used_gene_list: List[str],
                                 X: torch.Tensor,
                                 adj: torch.Tensor,
                                 cell_type_names: List[str],
                                 pathway_oi: str,
                                 fname: Optional[Union[str, Path]] = None,
                                 xticks_position='bottom',
                                 figsize=None,
                                 linewidths=0.6,
                                 ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """

    Plotting Heatmaps for specific pathway related genes, and which ligand receptors affect these genes,
    and these FCEs come from which cell types.

    Parameters
    ----------
    trained_MGC_model_list :
        A list of trained MGC model for generating the expression of one target gene.
    lr_df :
        The used preprocessed LR-gene dataframe, must contain the 'LR_pair' column.
    used_gene_list :
        List of used target gene names.
    X :
        The feature matrix used as the input of the trained_MGC_model_list.
    adj :
        The adjacency matrix used as the input of the trained_MGC_model_list.
    cell_type_names :
        List of cell-type names.
    pathway_oi :
        The pathway on interest. Should in the 'pathway_name' column of lr_df.
    fname :
        The output file name. If None, not save the figure.
    xticks_position :
        Plot xticks at 'top' or 'bottom'.
    figsize :
        Set the figsize.
    linewidths :
        Set the widths of inner lines of heatmap.

    Returns
    -------
    Three dataframes for the three subplots.

    """
    ranked_LR_df_list = []
    for MGC_model_list in tqdm(trained_MGC_model_list):
        ranked_LR_df_tmp = lr_rank_in_mgc(MGC_model_list, lr_df,
                                          plot_cluster=False, repeat_attention_scale=True, display=False)
        ranked_LR_df_list.append(ranked_LR_df_tmp.loc[lr_df.index, 'MGC_layer_attention'])

    all_target_rank = pd.concat(ranked_LR_df_list, axis=1)
    all_target_rank.columns = used_gene_list
    all_target_rank.index = lr_df['LR_Pair']

    used_ligands = lr_df[lr_df['pathway_name'] == pathway_oi]['LR_Pair']

    plot_df = pd.DataFrame(all_target_rank.loc[used_ligands].sum(0).sort_values(ascending=False).head(10))
    plot_df.columns = ['Attention']

    used_ligand_target_link = all_target_rank.loc[used_ligands, plot_df.index].T

    signal_from_cell_type_df = []
    for target in plot_df.index:
        target_id = used_gene_list.index(target)
        source_cell_type_list = []
        for lr in used_ligands:
            _ = fce_cell_type_network_plot(trained_MGC_model_list[target_id],
                                           lr_df, X, adj, cell_type_names,
                                           plot_lr=lr, hide_repeat_tqdm=True,
                                           edge_thres=0.2, display=False)
            source_cell_type_list.append(_[0].sum(1) * all_target_rank.loc[lr, target])
        signal_from_cell_type_df.append(pd.concat(source_cell_type_list, axis=1).mean(1))

    signal_from_cell_type_df = pd.DataFrame(signal_from_cell_type_df)
    signal_from_cell_type_df.index = plot_df.index

    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=(8, 3))

    all_width = 1 + round(len(used_ligands) / 3) + round(signal_from_cell_type_df.shape[1] / 3)
    grid = plt.GridSpec(1, all_width, wspace=1.5)

    if xticks_position == 'bottom':
        xticks_ha = 'right'
    else:
        xticks_ha = 'left'

    axes0 = plt.subplot(grid[0, 0])
    axes0 = sns.heatmap(plot_df, cmap='YlOrBr', linewidths=linewidths, vmin=0,
                        cbar_kws={"orientation": "horizontal", "shrink": 0.8})

    axes0.xaxis.set_ticks_position(xticks_position)
    plt.xticks(rotation=45, ha=xticks_ha)

    axes1 = plt.subplot(grid[0, 1:1 + round(len(used_ligands) / 3)])
    axes1 = sns.heatmap(used_ligand_target_link,
                        cmap='Purples', linewidths=linewidths, vmin=0,
                        cbar_kws={"orientation": "horizontal", "shrink": 0.8})
    axes1.xaxis.set_ticks_position(xticks_position)
    axes1.set_xlabel(None)
    plt.xticks(rotation=45, ha=xticks_ha)

    axes2 = plt.subplot(grid[0, 1 + round(len(used_ligands) / 3):all_width])
    axes2 = sns.heatmap(signal_from_cell_type_df, cmap='PuBu',
                        linewidths=linewidths, vmin=0,
                        cbar_kws={"orientation": "horizontal", "shrink": 0.8})
    axes2.xaxis.set_ticks_position(xticks_position)
    plt.xticks(rotation=45, ha=xticks_ha)

    if fname is not None:
        plt.savefig(fname)
    plt.show()

    return plot_df, used_ligand_target_link, signal_from_cell_type_df


def save_mgc_interpretation_for_all_target(trained_MGC_model_list: List[List[MGC_Model]],
                                           X: torch.Tensor,
                                           adj: torch.Tensor,
                                           used_gene_list: List[str],
                                           lr_df: pd.DataFrame,
                                           cell_type_names: List[str],
                                           figures_save_folder: Optional[Union[str, Path]],
                                           repeat_filter_num: int = 0,
                                           LR_pair_num_per_target: int = 15,
                                           heatmap_plot_lr_num: int = 15,
                                           project_name: str = 'all_target_results',
                                           save_fce_plot: bool = False,
                                           **kwargs,
                                           ) -> pd.DataFrame:
    """

    Save the interpretation figures of MGC models for multiple target genes.

    Parameters
    ----------
    trained_MGC_model_list :
        A list of trained MGC model for generating the expression of one target gene.
    X :
        The feature matrix used as the input of the trained_MGC_model_list.
    adj :
        The adjacency matrix used as the input of the trained_MGC_model_list.
    used_gene_list :
        List of used target gene names
    lr_df :
        The used preprocessed LR-gene dataframe, must contain the 'LR_pair' column.
    cell_type_names :
        List of cell-type names.
    figures_save_folder :
        Father folder of saving figures
    repeat_filter_num :
        The number of LR pair displayed in the output figure.
    LR_pair_num_per_target :
        The number of LR pair plotted in lr_rank_in_mgc plot and cell-type-level FCE network (one LR one network).
    heatmap_plot_lr_num:
        The number of LR pair plotted in heatmap.
    save_fce_plot:
        If False, not save FCE plot to save time.
    project_name :
        The name of project, such as 'BRCA_10x_generating_all_target_gene'.
    kwargs :
        Other parameters in 'fce_cell_type_network_plot' function.

    Returns
    -------
    Dataframe containing the interpretation results of MGC models for multiple target genes.

    """

    all_target_result = pd.DataFrame()
    row_id = 0

    for i in tqdm(range(len(used_gene_list))):

        target_i_results_PATH = os.path.join(figures_save_folder, project_name, used_gene_list[i])
        if not os.path.exists(os.path.join(target_i_results_PATH, 'SR_network')):
            os.makedirs(os.path.join(target_i_results_PATH, 'SR_network'))

        related_LR_df_MGC_result = lr_rank_in_mgc(trained_MGC_model_list[i], lr_df, repeat_filter_num=repeat_filter_num,
                                                  repeat_attention_scale=True, plot_lr_num=heatmap_plot_lr_num,
                                                  display=False, plot_cluster=False,
                                                  fname=os.path.join(target_i_results_PATH,
                                                                     used_gene_list[i] + '_LR_pair_importance.pdf'))

        LR_rank_in_model = [torch.argsort(abs(trained_MGC_model_list[i][model_i].mgc.layer_attention), descending=True)
                            for model_i in range(len(trained_MGC_model_list[i]))]

        for LR_id in range(LR_pair_num_per_target):
            all_target_result.loc[row_id, 'target_gene'] = used_gene_list[i]
            LR_Pair = related_LR_df_MGC_result.iloc[LR_id].LR_Pair
            all_target_result.loc[row_id, 'LR_Pair'] = LR_Pair
            all_target_result.loc[row_id, 'MGC_layer_attention'] = related_LR_df_MGC_result.iloc[
                LR_id].MGC_layer_attention

            attention_mean_rank = np.stack(
                [list(i).index(related_LR_df_MGC_result.index[LR_id]) for i in LR_rank_in_model])
            attention_mean_rank[attention_mean_rank > LR_pair_num_per_target] = LR_pair_num_per_target
            all_target_result.loc[row_id, 'attention_mean_rank'] = attention_mean_rank.mean()

            if save_fce_plot:
                fname = os.path.join(target_i_results_PATH, 'SR_network',
                                     used_gene_list[i] + '_' + LR_Pair + '_SR_network.pdf')
            else:
                fname = None
            SR_network, _ = fce_cell_type_network_plot(trained_MGC_model_list[i], lr_df, X, adj,
                                                       cell_type_names, plot_lr=LR_Pair,
                                                       display=False,
                                                       fname=fname,
                                                       hide_repeat_tqdm=True,
                                                       **kwargs)

            for index, row in SR_network.iteritems():
                edge_name = row.index + ':' + row.name
                for item in range(len(row)):
                    all_target_result.loc[row_id, edge_name[item]] = row[item]

            row_id = row_id + 1

    lr_df.index = lr_df['LR_Pair']
    col_name = all_target_result.columns.tolist()
    lr_info = lr_df.loc[list(all_target_result['LR_Pair']),].iloc[:, 1:]
    _ = [col_name.insert(4, i) for i in lr_info.columns[::-1]]
    all_target_result = all_target_result.reindex(columns=col_name)

    lr_info.index.name = 'index'
    all_target_result.loc[:, lr_info.columns] = lr_info.reset_index()[lr_info.columns]
    all_target_result.to_csv(os.path.join(figures_save_folder, project_name,
                                          'all_target_LR_attention_and_SR_network.csv'))

    return all_target_result
