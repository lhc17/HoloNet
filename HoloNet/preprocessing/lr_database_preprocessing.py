import numpy as np
import pandas as pd
from anndata import AnnData


def get_expressed_lr_df(lr_df: pd.DataFrame,
                        adata: AnnData,
                        expressed_proportion: float = 0.3,
                        ) -> pd.DataFrame:
    """\
    Filter the dataframe with pairwise ligand and receptor gene, requiring ligand and receptor genes to be expressed in a
    certain percentage of cells (or spots).
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
