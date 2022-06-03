import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .MGC_interpretation_plotting import lr_rank_in_mgc, fce_cell_type_network_plot
from ..predicting.MGC_model import MGC_Model


def save_mgc_interpretation_for_all_target(trained_MGC_model_list: List[MGC_Model],
                                           X: torch.Tensor,
                                           adj: torch.Tensor,
                                           used_gene_list: List[str],
                                           lr_df: pd.DataFrame,
                                           cell_type_names: List[str],
                                           figures_save_folder: Optional[Union[str, Path]],
                                           repeat_filter_num: int = 0,
                                           LR_pair_num_per_target: int = 15,
                                           project_name: str = 'all_target_results',
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
                                                  repeat_attention_scale=True, plot_lr_num=LR_pair_num_per_target,
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

            SR_network = fce_cell_type_network_plot(trained_MGC_model_list[i], lr_df, X, adj,
                                                    cell_type_names, plot_lr=LR_Pair,
                                                    display=False,
                                                    fname=os.path.join(target_i_results_PATH, 'SR_network',
                                                                       used_gene_list[
                                                                           i] + '_' + LR_Pair + '_SR_network.pdf'),
                                                    **kwargs)

            for index, row in SR_network.iteritems():
                edge_name = row.index + ':' + row.name
                for item in range(len(row)):
                    all_target_result.loc[row_id, edge_name[item]] = row[item]

            row_id = row_id + 1

    all_target_result.to_csv(os.path.join(figures_save_folder, project_name,
                                          'all_target_LR_attention_and_SR_network.csv'))

    return all_target_result
