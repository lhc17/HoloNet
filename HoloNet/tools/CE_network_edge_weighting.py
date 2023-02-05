from typing import List

import numpy as np
import pandas as pd
import torch
from anndata._core.anndata import AnnData
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gmean
from tqdm import tqdm


def compute_ce_tensor_connectomeDB(adata: AnnData,
                                   lr_df: pd.DataFrame,
                                   w_best: float,
                                   distinguish: bool = True,
                                   **kwargs,
                                   ) -> torch.Tensor:
    """\
    Calculate CE matrix for measuring the strength of communication between any pairs of cells, according to
    the edge weighting function.

    Parameters
    ----------
    adata
        Annotated data matrix.
    lr_df
        A preprocessed LR-gene dataframe.
        must contain three columns: 'Ligand_gene_symbol', 'Receptor_gene_symbol' and 'LR_pair'.
    w_best
        A distance parameter in edge weighting function controlling the covering region of ligands.
        'default_w_visium' function provides a recommended value of w_best.
    distinguish:
        If True, set the different w_best for secreted ligands and plasma-membrane-binding ligands.
    
    Returns
    -------
    A CE tensor (LR_pair_num * cell_num * cell_num)
    
    """

    dist_factor_tensor = distinguish_dist_factor_calculate(adata=adata,
                                                           lr_df=lr_df,
                                                           w_best=w_best,
                                                           distinguish=distinguish,
                                                           **kwargs, )

    expressed_ligand = lr_df.loc[:, 'Ligand_gene_symbol'].tolist()
    expressed_receptor = lr_df.loc[:, 'Receptor_gene_symbol'].tolist()

    expressed_ligand_tensor = get_gene_expr_tensor(adata, expressed_ligand)
    expressed_receptor_tensor = get_gene_expr_tensor(adata, expressed_receptor).permute(0, 2, 1)

    print(dist_factor_tensor.shape)
    print(expressed_ligand_tensor.shape)

    ce_tensor = expressed_ligand_tensor.mul(dist_factor_tensor).mul(expressed_receptor_tensor)
    ce_tensor = ce_tensor / ce_tensor.mean((1, 2)).unsqueeze(1).unsqueeze(2)

    return ce_tensor.to(torch.float32)


def compute_ce_tensor(used_interaction_db: pd.DataFrame,
                      w_best: float,
                      elements_expr_df_dict: dict,
                      adata: AnnData,
                      distinguish: bool = True,
                      anno_col: str = 'annotation',
                      two_class: dict = {'local': ['Cell-Cell Contact', 'ECM-Receptor'],
                                         'global': ['Secreted Signaling']},
                      dist_factor_secreted: bool = None,
                      ) -> torch.Tensor:
    """

    Calculate CE matrix for measuring the strength of communication between any pairs of cells, according to
    the edge weighting function.

    Parameters
    ----------
    used_interaction_db :
        A preprocessed LR-gene dataframe.
    w_best :
        A distance parameter in edge weighting function controlling the covering region of ligands.
        'default_w_visium' function provides a recommended value of w_best.
    elements_expr_df_dict :
        Metadata from 'elements_expr_df_calculate' function.
    adata :
        Annotated data matrix.
    distinguish :
        If True, set the different w_best for secreted ligands and plasma-membrane-binding ligands.
    anno_col :
        The column in used_interaction_db containing the proportion of ligand.
    two_class :
        Divide the LR pair into two class. The two class uses different w paramater.
    dist_factor_secreted :
        Calculated dist factor matrix for secreted LR pairs.

    Returns
    -------
    A CE tensor (LR_pair_num * cell_num * cell_num)

    """

    dist_factor = distinguish_dist_factor_calculate(adata=adata, lr_df=used_interaction_db, w_best=w_best,
                                                    distinguish=distinguish, anno_col=anno_col,
                                                    two_class=two_class)

    if dist_factor_secreted is None:
        dist_factor_secreted = distinguish_dist_factor_calculate(adata=adata, lr_df=used_interaction_db,
                                                                 w_best=w_best * 2, distinguish=False)

    AG_tensor = torch.tensor(np.array(elements_expr_df_dict['AG_expr'])).T.unsqueeze(1).mul(dist_factor_secreted).sum(2)
    AN_tensor = torch.tensor(np.array(elements_expr_df_dict['AN_expr'])).T.unsqueeze(1).mul(dist_factor_secreted).sum(2)

    l_tensor = torch.tensor(np.array(elements_expr_df_dict['ligand_expr'])).T.unsqueeze(1)
    AN_AG_tensor = (1 + AG_tensor) / (1 + AN_tensor)

    r_tensor = (torch.tensor(np.array(elements_expr_df_dict['co_and_receptor_expr'])).T * AN_AG_tensor).unsqueeze(1)
    ce_tensor = l_tensor.mul(dist_factor).mul(r_tensor.permute(0, 2, 1))

    return torch.tensor(ce_tensor, dtype=torch.float32)


def elements_expr_df_calculate(used_interaction_db: pd.DataFrame,
                               complex_db: pd.DataFrame,
                               cofactor_db: pd.DataFrame,
                               adata: AnnData,
                               ) -> dict:
    """

    Calculate the expression of elements, including ligand, receptor, co_and_receptor, AG and AN.
    Prepare for 'compute_ce_tensor' function

    Parameters
    ----------
    used_interaction_db :
        A preprocessed LR-gene dataframe.
    complex_db :
        A dataframe containing ligand or receptor complex.
    cofactor_db :
        A dataframe containing the cofactors for each LR pair.
    adata :
        Annotated data matrix.

    Returns
    -------
    A dictionary including ligand, receptor, co_and_receptor, AG and AN

    """

    co_and_receptor_df = pd.DataFrame()
    receptor_df = pd.DataFrame()
    ligand_df = pd.DataFrame()

    AG_df = pd.DataFrame()
    AN_df = pd.DataFrame()

    for interaction_term in tqdm(used_interaction_db.index):
        receptor_expr = lr_expr_calculate(interaction_term, used_interaction_db, complex_db, adata, l_or_r='receptor')
        receptor_df[interaction_term] = receptor_expr
        ligand_expr = lr_expr_calculate(interaction_term, used_interaction_db, complex_db, adata, l_or_r='ligand')
        ligand_df[interaction_term] = ligand_expr

        coA_expr = lr_expr_calculate(interaction_term, used_interaction_db, cofactor_db, adata,
                                     l_or_r='co_A_receptor', mean='mean')
        coI_expr = lr_expr_calculate(interaction_term, used_interaction_db, cofactor_db, adata,
                                     l_or_r='co_I_receptor', mean='mean')

        co_and_receptor_expr = receptor_expr * ((1 + coA_expr) / (1 + coI_expr)).T
        co_and_receptor_df[interaction_term] = co_and_receptor_expr

        AG_expr = lr_expr_calculate(interaction_term, used_interaction_db, cofactor_db, adata,
                                    l_or_r='agonist', mean='mean')
        AG_df[interaction_term] = AG_expr
        AN_expr = lr_expr_calculate(interaction_term, used_interaction_db, cofactor_db, adata,
                                    l_or_r='antagonist', mean='mean')
        AN_df[interaction_term] = AN_expr

    return {'co_and_receptor_expr': co_and_receptor_df,
            'receptor_expr': receptor_df,
            'ligand_expr': ligand_df,
            'AG_expr': AG_df,
            'AN_expr': AN_df}


def dist_factor_calculate(adata: AnnData,
                          w_best: float,
                          obsm_spatial_slot: str = 'spatial',
                          ) -> torch.Tensor:
    """

    Parameters
    ----------
    adata :
        Annotated data matrix.
    w_best :
        A distance parameter in edge weighting function controlling the covering region of ligands.
        'default_w_visium' function provides a recommended value of w_best.
    obsm_spatial_slot :
        The slot name storing the spatial position information of each spot。

    Returns
    -------
    A tensor describing the distance factor.

    """
    position_mat = adata.obsm[obsm_spatial_slot]
    dist_mat = squareform(pdist(position_mat, metric='euclidean'))
    dist_factor = dist_mat / w_best

    dist_factor = np.exp((-1) * dist_factor * dist_factor)
    dist_factor = torch.tensor(dist_factor).to(torch.float32)

    return dist_factor


'''
def distinguish_dist_factor_calculate(adata: AnnData,
                                      lr_df: pd.DataFrame,
                                      w_best: float,
                                      distinguish=False,
                                      ) -> torch.Tensor:
    if distinguish:
        w_best2 = w_best * 2
        dist_factor1 = dist_factor_calculate(adata=adata, w_best=w_best, )
        dist_factor2 = dist_factor_calculate(adata=adata, w_best=w_best2, )
        dist_factor_tensor = dist_factor1.repeat(lr_df.shape[0], 1, 1)
        secreted_index = lr_df[lr_df.Ligand_location == 'secreted'].index
        dist_factor_tensor[secreted_index, :, :] = dist_factor2
    else:
        dist_factor_tensor = dist_factor_calculate(adata, w_best=w_best, )

    return dist_factor_tensor
'''


def distinguish_dist_factor_calculate(adata: AnnData,
                                      lr_df: pd.DataFrame,
                                      w_best: float,
                                      distinguish=False,
                                      anno_col='Ligand_location',
                                      two_class={'local': ['plasma membrane'],
                                                 'global': ['secreted']},
                                      obsm_spatial_slot='spatial',
                                      ) -> torch.Tensor:
    if distinguish:
        w_best2 = w_best * 2
        dist_factor1 = dist_factor_calculate(adata=adata, w_best=w_best, obsm_spatial_slot=obsm_spatial_slot)
        dist_factor2 = dist_factor_calculate(adata=adata, w_best=w_best2, obsm_spatial_slot=obsm_spatial_slot)

        position_mat = adata.obsm[obsm_spatial_slot]
        dist_mat = squareform(pdist(position_mat, metric='euclidean'))
        local_spot = np.where(dist_mat < dist_mat[dist_mat > 0].min() * 1.05)
        dist_factor_new = torch.zeros([dist_factor1.shape[0], dist_factor1.shape[1]])
        dist_factor_new[local_spot[0], local_spot[1]] = dist_factor1[local_spot[0], local_spot[1]]

        dist_factor_tensor = dist_factor_new.repeat(lr_df.shape[0], 1, 1)

        secreted_index = np.where(lr_df[anno_col].isin(two_class['global']))[0]
        dist_factor_tensor[secreted_index, :, :] = dist_factor2
    else:
        dist_factor_tensor = dist_factor_calculate(adata, w_best=w_best, obsm_spatial_slot=obsm_spatial_slot)

    return dist_factor_tensor


def get_gene_expr_tensor(adata: AnnData,
                         gene_name: List[str],
                         ) -> torch.Tensor:
    gene_expr_mat = adata[:, gene_name].X.toarray().astype(np.float32)
    gene_expr_tensor = torch.tensor(np.expand_dims(gene_expr_mat, 2)).permute(1, 2, 0)

    return gene_expr_tensor


def lr_expr_calculate(interaction_term, interaction_db, complex_db, adata, l_or_r, mean='gmean'):
    lr_term = interaction_db.loc[interaction_term, l_or_r]
    if str(lr_term) == 'nan':
        return np.zeros([adata.shape[0]])

    if lr_term in complex_db.index:
        lr_list = list(complex_db.loc[lr_term])
        lr_list = [x for x in lr_list if str(x) != 'nan']
    else:
        lr_list = [lr_term]

    if mean == 'gmean':
        if len(set(lr_list) - set(adata.var_names)) != 0:
            lr_expr = np.zeros([adata.shape[0]])
        else:
            lr_expr = gmean(np.array(adata[:, lr_list].to_df()).T)
    else:
        used_lr = list(set(lr_list).intersection(set(adata.var_names)))
        lr_expr = np.array(adata[:, used_lr].to_df()).T.mean(0)
    return lr_expr
