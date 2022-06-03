import json
from pathlib import Path
from typing import Union, Optional

import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.image import imread


def read_visium(
        path: Union[str, Path],
        genome: Optional[str] = None,
        *,
        mtx_directory: Optional[str] = None,
        count_file: Optional[str] = None,
        provided_adata: Optional[AnnData] = None,
        library_id: Optional[str] = None,
        load_images: Optional[bool] = True,
        source_image_path: Optional[Union[str, Path]] = None,
) -> AnnData:
    """
    Similar with sc.read_visium. Add methods for load the 10x mtx folder as a spatial Anndata, or add
    spatial information into a provided Anndata.

    Parameters
    ----------
    provided_adata
        A provided Anndata. The function add spatial information into the adata.
    mtx_directory
        The path for the 10x mtx folder.
    Other Parameters
        Same as sc.read_visium

    Returns
    -------
    Annotated data matrix with spatial information

    """
    path = Path(path)

    if mtx_directory is not None:
        adata = sc.read_10x_mtx(path / mtx_directory)
    if count_file is not None:
        adata = sc.read_10x_h5(path / count_file, genome=genome)
    if provided_adata is not None:
        adata = provided_adata

    adata.uns["spatial"] = dict()

    from h5py import File

    if count_file is not None:
        with File(path / count_file, mode="r") as f:
            attrs = dict(f.attrs)
        if library_id is None:
            library_id = str(attrs.pop("library_ids")[0], "utf-8")
    else:
        library_id = 'a'

    adata.uns["spatial"][library_id] = dict()

    if load_images:
        files = dict(
            tissue_positions_file=path / 'spatial/tissue_positions_list.csv',
            scalefactors_json_file=path / 'spatial/scalefactors_json.json',
            hires_image=path / 'spatial/tissue_hires_image.png',
            lowres_image=path / 'spatial/tissue_lowres_image.png',
        )

        adata.uns["spatial"][library_id]['images'] = dict()
        for res in ['hires', 'lowres']:
            try:
                adata.uns["spatial"][library_id]['images'][res] = imread(
                    str(files[f'{res}_image'])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]['scalefactors'] = json.loads(
            files['scalefactors_json_file'].read_bytes()
        )
        if count_file is not None:
            adata.uns["spatial"][library_id]["metadata"] = {
                k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
                for k in ("chemistry_description", "software_version")
                if k in attrs
            }

        # read coordinates
        positions = pd.read_csv(files['tissue_positions_file'], header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']

        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm['spatial'] = adata.obs[
            ['pxl_row_in_fullres', 'pxl_col_in_fullres']
        ].to_numpy()
        adata.obs.drop(
            columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'],
            inplace=True,
        )

        # put image path in uns
        if source_image_path is not None:
            # get an absolute path
            source_image_path = str(Path(source_image_path).resolve())
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                source_image_path
            )

    return adata
