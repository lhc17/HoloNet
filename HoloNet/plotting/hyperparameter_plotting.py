from pathlib import Path
from typing import Optional, Union

import numpy as np
import scanpy as sc
from anndata._core.anndata import AnnData
from matplotlib.colors import Colormap

from ..tools.CE_network_edge_weighting import dist_factor_calculate


def select_w(adata: AnnData,
             w_best: float,
             size: float = 1.4,
             alpha: float = 0.7,
             cmap: Union[Colormap, str] = 'Spectral_r',
             data_type: str = 'Visium',
             fname: Optional[Union[str, Path]] = None,
             **kwargs,
             ):
    """
    
    Plot the covering spatial region of ligands derived from one spot.

    Parameters
    ----------
    adata :
        Annotated data matrix.
    w_best :
        A distance parameter in edge weighting function controlling the covering region of ligands.
        'default_w_visium' function provides a recommended value of w_best.
    size :
        The size of each spot, see sc.pl.spatial in Scanpy.
    alpha :
        The alpha value of each spot.
    cmap :
        The color map of feature value of each spot, defaultly as 'Spectral_r'
    data_type :
        Select in Visium or SeqFISH
    fname :
        The output file name. If None, not save the figure. Note that not add path name, sc.pl.spatial will add 'show'
        before the file name.
    kwargs :
        Other parameters in scanpy.pl.spatial or scanpy.pl.embedding.

    """
    dist_factor = dist_factor_calculate(adata=adata,
                                        w_best=w_best)

    dist_factor_adata = AnnData(np.array(dist_factor), obs=adata.obs,
                                uns=adata.uns, obsm=adata.obsm, dtype='float64')
    tmp = dist_factor_calculate(adata, w_best=w_best).sum(1)
    display_pos = np.where(tmp == tmp.max())[0][0].astype('str')

    if data_type == 'Visium':
        sc.pl.spatial(dist_factor_adata, color=[display_pos], size=size, cmap=cmap, alpha=alpha, title='',
                      save=fname, **kwargs)
    if data_type == 'SeqFISH':
        sc.pl.embedding(dist_factor_adata, basis="spatial", color=[display_pos], size=size, **kwargs)
