import numpy as np
from anndata._core.anndata import AnnData
from scipy.spatial.distance import squareform, pdist


def default_w_visium(adata: AnnData,
                     min_cell_distance: int = 100,
                     cover_distance: int = 255,
                     obsm_spatial_slot: str = 'spatial',
                     ) -> float:
    """\
    Calculate a recommended value for the distance parameter in the edge weighting function.

    Parameters
    ----------
    adata
        Annotated data matrix.
    min_cell_distance
        The min distance between spots is 100 micrometers in 10x Visium technology.
    cover_distance
        Ligands cover a region with 255 micrometers diameter at a fixed concentration by default.
        The diameter of sender spot is 55 micrometers, and the ligands spread 100 micrometers.
    obsm_spatial_slot
        The slot name storing the spatial position information of each spot

    Returns
    -------
    A integer for a recommended value for the distance parameter in the edge weighting function.
    """

    position_mat = adata.obsm[obsm_spatial_slot]
    dist_mat = squareform(pdist(position_mat, metric='euclidean'))

    # ligands cover 255 micrometers by default, and the min value of distance between spot is 100 micrometers
    w_best = cover_distance * (dist_mat[dist_mat > 0].min() / min_cell_distance) / np.sqrt(np.pi)

    return w_best
