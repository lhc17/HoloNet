from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from anndata._core.anndata import AnnData

from .base_plot import feature_plot
from ..colorSchemes import color_sheet
from ..predicting.MGC_model import MGC_Model
from ..predicting.MGC_training import get_mgc_result


def plot_mgc_result(trained_MGC_model_list: List[MGC_Model],
                    adata: AnnData,
                    X: torch.Tensor,
                    adj: torch.Tensor,
                    hide_repeat_tqdm: bool = False,
                    **kwarg,
                    ) -> torch.Tensor:
    """

    Plot the generated expression profile of the target gene.

    Parameters
    ----------
    trained_MGC_model_list :
        A list of trained MGC model for generating the expression of one target gene.
    adata :
        Annotated data matrix.
    X :
        The feature matrix used as the input of the trained_MGC_model_list.
    adj :
        The adjacency matrix used as the input of the trained_MGC_model_list.
    hide_repeat_tqdm :
        If true, hide the tqdm for getting the result of repeated training.
    kwarg :
        See in 'feature_plot' function.

    Returns
    -------
    The generated expression profile of the target gene (cell_num * 1).

    """
    result = get_mgc_result(trained_MGC_model_list, X, adj, hide_repeat_tqdm=hide_repeat_tqdm, )
    feature_plot(result, adata, ['Generated expression'], 'Generated expression', **kwarg)

    return result


def find_genes_linked_to_ce(predicted_expr_type_MGC_df: pd.DataFrame,
                            predicted_expr_only_type_df: pd.DataFrame,
                            used_gene_list: List[str],
                            target_all_gene_expr: torch.Tensor,
                            plot_gene_list: Optional[List[str]] = None,
                            fname: Optional[Union[str, Path]] = None,
                            display=True,
                            **kwargs,
                            ) -> pd.DataFrame:
    """

    Plot the correlation of MGC model results with ground truth,
    and the results of model only using cell-type with ground truth.

    Parameters
    ----------
    predicted_expr_type_MGC_df :
        A dataframe for the generated expression of multiple genes in each cell from MGC model using CE network and
        cell-type information, as the output of 'get_mgc_result_for_multiple_targets' function.
    predicted_expr_only_type_df :
        A dataframe for the generated expression of multiple genes in each cell from model only using cell-type
        information, as the output of 'get_mgc_result_for_multiple_targets' function.
    used_gene_list :
        List of used target gene names
    target_all_gene_expr :
        The scaled expression tensor of target genes (cell_num * target_gene_num), from 'get_gene_expr' function.
    plot_gene_list :
        The genes will be pointed out in the output figure.
    fname :
        The output file name. If None, not save the figure.
    display :
        If False, not plot the figure.
    kwargs :
        See in plt.scatter

    Returns
    -------
    The correlation of MGC model results with ground truth,
    and the results of model only using cell-type with ground truth.


    """

    target_gene_num = len(used_gene_list)

    type_MGC_coef_list = [np.corrcoef(np.array(predicted_expr_type_MGC_df)[:, i],
                                      np.array(target_all_gene_expr)[i, :])[0, 1] for i in range(target_gene_num)]
    only_type_coef_list = [np.corrcoef(np.array(predicted_expr_only_type_df)[:, i],
                                       np.array(target_all_gene_expr)[i, :])[0, 1] for i in range(target_gene_num)]

    only_type_vs_MGC = np.stack([np.stack(only_type_coef_list), np.stack(type_MGC_coef_list)]).T
    only_type_vs_MGC = pd.DataFrame(only_type_vs_MGC, index=used_gene_list)
    only_type_vs_MGC.columns = ['only_cell_type', 'cell_type_and_MGC']
    only_type_vs_MGC['difference'] = only_type_vs_MGC.cell_type_and_MGC - only_type_vs_MGC.only_cell_type
    only_type_vs_MGC = only_type_vs_MGC.sort_values(by='difference', ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plot
    ax.scatter(only_type_vs_MGC["only_cell_type"],
               only_type_vs_MGC["cell_type_and_MGC"],
               c=color_sheet['dark_blue_qualitative'],
               marker='o',
               edgecolors='black',
               alpha=1, **kwargs)

    low_lim = only_type_vs_MGC.loc[:, ['only_cell_type', 'cell_type_and_MGC']].min().min() - 0.05
    high_lim = only_type_vs_MGC.loc[:, ['only_cell_type', 'cell_type_and_MGC']].max().max() + 0.05

    ax.set_xlim(low_lim, high_lim)
    ax.set_ylim(low_lim, high_lim)

    ax.set_xlabel(r'Pearson correlation (without MGC)')
    ax.set_ylabel(r'Pearson correlation (with MGC)')
    ax.plot([low_lim, high_lim], [low_lim, high_lim], color=color_sheet['red_qualitative'], linestyle='dashed')

    if plot_gene_list is not None:
        for target_gene in plot_gene_list:
            plt.annotate(r'{}'.format(target_gene),
                         xy=(only_type_vs_MGC.loc[target_gene, 'only_cell_type'],
                             only_type_vs_MGC.loc[target_gene, 'cell_type_and_MGC']), xycoords='data',
                         xytext=(-40, +50), textcoords='offset points', fontsize=8,
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"))

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', )
    if display:
        plt.show()
    plt.close()

    return only_type_vs_MGC


def single_view_mgc_coef_plot(lr_df: pd.DataFrame,
                              CCI_coef_list: List[float],
                              red_line: Optional[float] = None,
                              yellow_line: Optional[float] = None,
                              green_line: Optional[float] = None,
                              plot_lr_list: Optional[List[str]] = None,
                              fname: Optional[Union[str, Path]] = None,
                              ) -> pd.DataFrame:
    """

    Parameters
    ----------
    lr_df :
    CCI_coef_list :
    red_line :
    yellow_line :
    green_line :
    plot_lr_list :
    fname :

    Returns
    -------

    """

    lr_df_tmp = lr_df.copy()
    lr_df_tmp['coef'] = CCI_coef_list
    lr_df_tmp = lr_df_tmp.sort_values('coef', ascending=True).reset_index()

    max_value, min_value = np.stack(CCI_coef_list).max(), np.stack(CCI_coef_list).min()

    plt.figure(figsize=(5, 4))
    plt.scatter([i for i in range(lr_df_tmp.shape[0])],
                lr_df_tmp['coef'],
                s=8)

    if red_line is not None:
        plt.axhline(red_line, linestyle='--', color='r')
        max_value, min_value = max(max_value, red_line), min(min_value, red_line)

    if yellow_line is not None:
        plt.axhline(yellow_line, linestyle='--', color='y')
        max_value, min_value = max(max_value, yellow_line), min(min_value, yellow_line)

    if green_line is not None:
        plt.axhline(green_line, linestyle='--', color='g')
        max_value, min_value = max(max_value, green_line), min(min_value, green_line)

    plt.ylim(min_value - 0.05, max_value + 0.05)
    
    if plot_lr_list is not None:
        for target_lr in plot_lr_list:
            target_lr_id = list(lr_df_tmp['LR_Pair']).index(target_lr)
            plt.annotate(r'{}'.format(target_lr),
                         xy=(target_lr_id,
                             lr_df_tmp.iloc[target_lr_id]['coef']), xycoords='data',
                         xytext=(-40, +50), textcoords='offset points', fontsize=8,
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"))

    if fname is not None:
        plt.savefig(fname)
    plt.show()

    return lr_df_tmp
