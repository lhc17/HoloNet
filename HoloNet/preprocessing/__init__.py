from .adata_loading import read_visium
from .lr_database_preprocessing import get_expressed_lr_df



"""Example datasets"""

from anndata._core.anndata import AnnData
import scanpy as sc
import pandas as pd


def load_brca_visium_10x() -> AnnData:
    """\
    
    Load the example dataset
    From the 10x Genomics website (https://www.10xgenomics.com/resources/datasets)
    From fresh frozen Invasive Ductal Carcinoma breast tissue (Block A Section 1)
    Profiled the expression of 24,923 genes in 3,798 spots
    
    Our preprocessing:
        Exclude spots with less than 500 UMIs and genes expressed in less than 3 spots 
        Normalize the expression matrix with the LogNormalize method in Seurat. 
        Annotate the cell types by label transfer (the TransferData function in Seurat) 
            with single-cell breast cancer dataset GSE118390 as reference dataset. 
        Deconvolution results stored at adata.obsm['predicted_cell_type']
        Cell-type label (the max value) stored at adata.obs.cell_type
        
    Returns
    -------
    Annotated data matrix.
    
    """
    
    url = "https://cloud.tsinghua.edu.cn/f/dd941f0d12214e6583bb/?dl=1"
    filename = sc.settings.datasetdir / "BRCA_Visium_10x_tmp.h5ad"
    adata = sc.read(filename, backup_url=url)
    return adata


def load_lr_df() -> pd.DataFrame:
    """\
    Load the provided dataframe with the information on ligands and receptors.
    
    Returns
    -------
    The LR-gene dataframe.
    
    """
    
    LR_pair_database_url = 'https://cloud.tsinghua.edu.cn/f/bb1080f2c5ba49cd815b/?dl=1'
    connectomeDB = pd.read_csv(LR_pair_database_url, encoding='Windows-1252')
    used_connectomeDB = connectomeDB.loc[:,['Ligand gene symbol','Receptor gene symbol','Ligand location']]
    used_connectomeDB.columns = ['Ligand_gene_symbol','Receptor_gene_symbol','Ligand_location']
    
    return used_connectomeDB


