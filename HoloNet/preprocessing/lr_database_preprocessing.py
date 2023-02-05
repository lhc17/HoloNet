import os

import numpy as np
import pandas as pd
from anndata._core.anndata import AnnData
from tqdm import tqdm

from ..tools.CE_network_edge_weighting import lr_expr_calculate


def load_lr_df_connectomeDB() -> pd.DataFrame:
    """\
    Load the provided dataframe with the information on ligands and receptors.
    
    Returns
    -------
    The LR-gene dataframe.
    
    """
    LR_pair_database_path = './data/ConnectomeDB2020.csv'
    if os.path.exists(LR_pair_database_path):
        connectomeDB = pd.read_csv(LR_pair_database_path, encoding='Windows-1252')
    else:
        LR_pair_database_url = 'https://cloud.tsinghua.edu.cn/f/bb1080f2c5ba49cd815b/?dl=1'
        connectomeDB = pd.read_csv(LR_pair_database_url, encoding='Windows-1252')

    used_connectomeDB = connectomeDB.loc[:, ['Ligand gene symbol', 'Receptor gene symbol', 'Ligand location']]
    used_connectomeDB.columns = ['Ligand_gene_symbol', 'Receptor_gene_symbol', 'Ligand_location']

    return used_connectomeDB


def load_lr_df(human_or_mouse: str = 'human',
               ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
        Load the provided dataframe with the information on ligands and receptors.

    Parameters
    ----------
    human_or_mouse :
        Select in 'human' and 'mouse', depending on your dataset.

    Returns
    -------
    Three dataframe: interaction_db, cofactor_db, complex_db

    """

    if human_or_mouse == 'human':
        cofactor_db_url = 'https://cloud.tsinghua.edu.cn/f/353d559dbf9047d6b422/?dl=1'
        cofactor_db = pd.read_csv(cofactor_db_url, index_col=0)
        complex_db_url = 'https://cloud.tsinghua.edu.cn/f/ad1152636d0b40ba9cb1/?dl=1'
        complex_db = pd.read_csv(complex_db_url, index_col=0)
        interaction_db_url = 'https://cloud.tsinghua.edu.cn/f/a0e4a9254de64a5dacd8/?dl=1'
        interaction_db = pd.read_csv(interaction_db_url, index_col=0)
    elif human_or_mouse == 'mouse':
        cofactor_db_url = 'https://cloud.tsinghua.edu.cn/f/9e0e850512ea47e6b9a4/?dl=1'
        cofactor_db = pd.read_csv(cofactor_db_url, index_col=0)
        complex_db_url = 'https://cloud.tsinghua.edu.cn/f/670109c134824cf1a8d7/?dl=1'
        complex_db = pd.read_csv(complex_db_url, index_col=0)
        interaction_db_url = 'https://cloud.tsinghua.edu.cn/f/5b0482f207244150a095/?dl=1'
        interaction_db = pd.read_csv(interaction_db_url, index_col=0)
    else:
        raise OSError('Please select human_or_mouse parameter in \'human\' and \'mouse\'')

    return interaction_db, cofactor_db, complex_db


def get_expressed_lr_df_connectomeDB(lr_df: pd.DataFrame,
                                     adata: AnnData,
                                     expressed_proportion: float = 0.3,
                                     ) -> pd.DataFrame:
    """\
    Filter the dataframe with pairwise ligand and receptor gene, requiring ligand and receptor genes to be expressed
    in a certain percentage of cells (or spots).
    And generate the 'LR_pair' column as the used names of ligand–receptor pairs in following workflow.
    
    Parameters
    ----------
    lr_df
        A pandas dataframe, must contain two columns: 'Ligand_gene_symbol' and 'Receptor_gene_symbol'.
    adata
        Annotated data matrix.
    expressed_proportion
        The percentage of cells required to express ligand and receptor genes in at least.
    
    Returns
    -------
    A preprocessed LR-gene dataframe.
    """

    expr_proportion_pass = np.array((np.sum(adata.X > 0, axis=0) > (adata.shape[0] * expressed_proportion))).squeeze()
    expressed_gene = adata.var.iloc[[i for i, x in enumerate(expr_proportion_pass) if x]].index.tolist()

    expressed_lr_df = lr_df \
        .loc[lambda x: x['Ligand_gene_symbol'].isin(expressed_gene)] \
        .loc[lambda x: x['Receptor_gene_symbol'].isin(expressed_gene)]

    expressed_lr_df = expressed_lr_df.reset_index(drop=True)
    expressed_lr_df['LR_Pair'] = [expressed_lr_df.Ligand_gene_symbol[i] + ':' +
                                  expressed_lr_df.Receptor_gene_symbol[i]
                                  for i in range(len(expressed_lr_df))]

    return expressed_lr_df


def get_expressed_lr_df(interaction_db, complex_db, adata, expressed_prop=0.15):
    """\
    Filter the dataframe with pairwise ligand and receptor gene, requiring ligand and receptor genes to be expressed in
    a certain percentage of cells (or spots).
    And generate the 'LR_pair' column as the used names of ligand–receptor pairs in following workflow.
    
    Parameters
    ----------
    interaction_db
        from cellchatDB
    complex_db
        from cellchatDB
    adata
        Annotated data matrix.
    expressed_prop
        The percentage of cells required to express ligand and receptor genes in at least.
    
    Returns
    -------
    A preprocessed LR-gene dataframe.
    """

    receptor_df = pd.DataFrame()
    ligand_df = pd.DataFrame()
    for interaction_term in tqdm(interaction_db.index):
        receptor_expr = lr_expr_calculate(interaction_term, interaction_db, complex_db, adata, l_or_r='receptor')
        receptor_df[interaction_term] = receptor_expr
        ligand_expr = lr_expr_calculate(interaction_term, interaction_db, complex_db, adata, l_or_r='ligand')
        ligand_df[interaction_term] = ligand_expr

    r_pass = np.where(np.sum(receptor_df > 0, axis=0) > adata.shape[0] * expressed_prop)[0]
    l_pass = np.where(np.sum(ligand_df > 0, axis=0) > adata.shape[0] * expressed_prop)[0]
    used_interaction_db = interaction_db.iloc[list(set(r_pass).intersection(l_pass))]

    used_interaction_db['LR_Pair'] = used_interaction_db['interaction_name_2'].str.replace(' - ', ':')
    # used_interaction_db = used_interaction_db.rename(columns = {'interaction_name':'LR_Pair'})

    return used_interaction_db
