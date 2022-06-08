from typing import List, Tuple

import pandas as pd
import torch
from anndata._core.anndata import AnnData
from tqdm import tqdm

from .MGC_model import MGC_Model
from .MGC_training import mgc_repeat_training, get_mgc_result, get_device


def mgc_training_for_multiple_targets(X: torch.Tensor,
                                      adj: torch.Tensor,
                                      target_all_gene_expr: torch.Tensor,
                                      repeat_num: int = 5,
                                      device: str = 'cpu',
                                      **kwargs,
                                      ) -> Tuple[List[List[MGC_Model]], List[List[MGC_Model]]]:
    """

    Using cell-type tensor and normalized adjancency matrix as the inputs, repeated training GNN to generate the
    expression of multiple target genes. Also generate the expression only using cell-type information.

    Parameters
    ----------
    X :
        A tensor (cell_num * cell_type_num) with cell-type information. derived from 'get_continuous_cell_type_tensor'
        or 'get_one_hot_cell_type_tensor' function.
    adj :
        A normalized adjancency matrix derived from 'adj_normalize' function.
    target_all_gene_expr :
        The scaled expression tensor of target genes (cell_num * target_gene_num), from 'get_gene_expr' function.
    repeat_num :
        The number of repeated training, defaultly as 50.
    use_gpu :
        If true, model will be trained in GPU when GPU is available.
    kwargs :
        Other training hyperparameter in 'mgc_repeat_training' function.

    Returns
    -------
    Two list of multiple (repeated training) trained MGC model for generating the expression of target genes, one using
    MGC with cell-type information and CE tensor, one only with cell-type information.

    """

    trained_MGC_model_list_only_type_list = []
    trained_MGC_model_list_type_MGC_list = []
    
    device = get_device(device)
    for i in tqdm(range(target_all_gene_expr.shape[0])):
        target_gene_expr = target_all_gene_expr[i, :]

        trained_MGC_model_list_only_type_for_one_target = mgc_repeat_training(X, adj, target_gene_expr,
                                                                              repeat_num=repeat_num,
                                                                              only_cell_type=True,
                                                                              hide_repeat_tqdm=True,
                                                                              device=device,
                                                                              **kwargs, )

        trained_MGC_model_list_type_MGC_for_one_target = mgc_repeat_training(X, adj, target_gene_expr,
                                                                             repeat_num=repeat_num,
                                                                             only_cell_type=False,
                                                                             hide_repeat_tqdm=True,
                                                                             device=device,
                                                                             **kwargs, )

        trained_MGC_model_list_only_type_list.append(trained_MGC_model_list_only_type_for_one_target)
        trained_MGC_model_list_type_MGC_list.append(trained_MGC_model_list_type_MGC_for_one_target)

    return trained_MGC_model_list_only_type_list, trained_MGC_model_list_type_MGC_list


def get_mgc_result_for_multiple_targets(trained_multi_MGC_model_list: List[List[MGC_Model]],
                                        X: torch.Tensor,
                                        adj: torch.Tensor,
                                        used_gene_list: List[str],
                                        adata: AnnData,
                                        device: str = 'cpu',
                                        ) -> pd.DataFrame:
    """

    Run the trained MGC model and get the generated expression profile of target genes.

    Parameters
    ----------
    trained_multi_MGC_model_list :
        A list of multiple (repeated training) trained MGC model for generating the expression of target genes.
    X :
        The feature matrix used as the input of the trained_MGC_model_list.
    adj :
        The adjacency matrix used as the input of the trained_MGC_model_list.
    used_gene_list :
        The list of target gene names used as the target of the trained_MGC_model_list.
    adata :
        Annotated data matrix.

    Returns
    -------
        A dataframe for the generated expression of multiple genes in each cell.

    """
    
    device = get_device(device)
    
    predicted_expr_list = []
    target_gene_num = len(trained_multi_MGC_model_list)

    for i in tqdm(range(target_gene_num)):
        trained_MGC_model_tmp = trained_multi_MGC_model_list[i]
        predicted_expr = get_mgc_result(trained_MGC_model_tmp, X, adj, 
                                        device=device, hide_repeat_tqdm=True)
        predicted_expr_list.append(predicted_expr.squeeze(1))

    predicted_expr_mat = torch.stack(predicted_expr_list)
    predicted_expr_df = pd.DataFrame(predicted_expr_mat.detach().numpy().T,
                                     columns=used_gene_list, index=adata.obs_names)

    return predicted_expr_df
