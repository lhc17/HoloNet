from pathlib import Path
from typing import Union, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from anndata._core.anndata import AnnData
from tqdm import tqdm


def get_gene_expr(adata: AnnData,
                  lr_df: Optional[pd.DataFrame] = None,
                  complex_db: Optional[pd.DataFrame] = None,
                  gene_list_on_interest: Optional[List[str]] = None,
                  min_mean: float = 0.05,
                  min_disp: float = 0.5,
                  max_sparse: float = 0.5,
                  remove_lr_gene: bool = True,
                  ) -> Tuple[torch.Tensor, List[str]]:
    """\
    Filter out the genes with too low expression levels or too low dispersions.
    Filter out the genes expressed in too fewer cells (or spots).
    Filter out the mitochondria genes and ligand (or receptor) genes.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    lr_df
        A pandas dataframe, must contain three columns: 'Ligand_gene_symbol', 'Receptor_gene_symbol' and 'LR_pair'.
    gene_list_on_interest:
        User provided gene list for the predefined genes on interest.
    min_mean:
        The minimum value of the mean expression level of filtered genes. (provided in adata.var.means)
    min_disp:
        The minimum value of the dispersions of filtered gene expressions. (provided in adata.var.dispersions_norm)
    max_sparse:
        The percentage of cells required to express the filtered genes.
    remove_lr_gene
        If True, filter out the ligand (or receptor) genes, to avoid data Leakage in predicting target gene expression
        using ligand–receptor pairs.
    
    Returns
    -------
    target_all_gene_expr
        Generated target gene expression matrix (cell by gene, scaled to make the maximum expression level of
        each target gene become 1)
    used_gene_list
        The filtered target gene list for the following workflow.
        
    """

    if ('dispersions_norm' not in adata.var.columns) or ('means' not in adata.var.columns):
        raise Exception(
            'Not detect the highly variable genes info in adata.var, please run sc.pp.highly_variable_genes before.')

    hvgenes = adata.var[adata.var.dispersions_norm > min_disp][adata.var.means > min_mean].index
    mt_genes = [adata.var.index[i] for i in range(adata.shape[1]) if adata.var.index[i].split('-')[0] == 'MT']

    non_zero_exp_genes = (np.array(adata.to_df()) != 0).sum(0)
    adata.var['non_zero_exp'] = non_zero_exp_genes / adata.shape[0]
    non_zero_exp_genes = adata.var.index[np.where(non_zero_exp_genes > adata.shape[0] * max_sparse)[0]]

    if gene_list_on_interest is not None:
        used_gene_list = gene_list_on_interest
    else:
        l_col = list(set(['ligand', 'Ligand_gene_symbol']).intersection(set(lr_df.columns)))[0]
        r_col = list(set(['receptor', 'Receptor_gene_symbol']).intersection(set(lr_df.columns)))[0]
        
        lr_list = np.array(complex_db).reshape(complex_db.shape[0] * complex_db.shape[1])
        lr_list = np.unique([x for x in lr_list if str(x) != 'nan'])

        if remove_lr_gene:
            used_gene_list = list(set(non_zero_exp_genes).intersection(set(hvgenes)) - set(mt_genes)
                                  - set(lr_df[l_col]) - set(lr_df[r_col]) - set(lr_list))
        else:
            used_gene_list = list(set(non_zero_exp_genes).intersection(set(hvgenes)) - set(mt_genes))

    target_all_gene_expr = adata[:, np.array(used_gene_list)].X.toarray()
    target_all_gene_expr = torch.Tensor(target_all_gene_expr / target_all_gene_expr.max(0)).T

    return target_all_gene_expr, used_gene_list


def get_one_case_expr(target_all_expr: torch.Tensor,
                      cases_list: List[str],
                      used_case_name: str,
                      ) -> torch.Tensor:
    """

    Get a cell_num*1 tensor representing the scaled expression profile of one gene, using as the target of GNN.

    Parameters
    ----------
    target_all_expr :
        Expression matrix of all target genes (from 'get_gene_expr' function).
    cases_list :
        All target gene names (from 'get_gene_expr' function).
    used_case_name :
        The gene name for the output expression vector.

    Returns
    -------
    cell_num*1 tensor representing the scaled expression profile of one gene

    """
    target = target_all_expr[cases_list.index(used_case_name)]
    target = target / target.max()
    target = target.to(torch.float32)
    target = target.unsqueeze(1)

    return target


def get_geneset_expr(geneset_PATH: Union[str, Path],
                     adata: AnnData,
                     ) -> Tuple[torch.Tensor, List[List[str]], List[str]]:
    """

    Parameters
    ----------
    geneset_PATH :
    adata :

    Returns
    -------

    """
    file_handler = open(geneset_PATH, "r")
    list_of_lines = file_handler.readlines()
    file_handler.close()

    set_genes_list = []
    set_name_list = []
    for line in list_of_lines:
        set_genes_list.append(line.strip().split('\t')[2:])
        set_name_list.append(line.strip().split('\t')[0])

    geneset_expr_list = []
    for set_gene in tqdm(set_genes_list):
        used_set_gene = [i for i in set_gene if i in adata.var.index]
        geneset_expr_list.append(torch.tensor(adata[:, used_set_gene].X.sum(1)).squeeze(1))
    target_all_geneset_expr = torch.stack(geneset_expr_list)
    target_all_geneset_expr = torch.tensor(target_all_geneset_expr.T / target_all_geneset_expr.numpy().max(1)).T

    return target_all_geneset_expr, set_genes_list, set_name_list
