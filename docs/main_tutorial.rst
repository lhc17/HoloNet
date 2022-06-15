Tutorials
===========


Tutorial 1
^^^^^^^^^^^

.. code:: ipython3

    import HoloNet as hn
    
    import os
    import pandas as pd
    import numpy as np
    import scanpy as sc
    import matplotlib.pyplot as plt
    import torch
    
    import warnings
    warnings.filterwarnings('ignore')
    hn.set_figure_params(tex_fonts=False)
    sc.settings.figdir = './figures/'

.. code:: ipython3

    adata = hn.pp.load_brca_visium_10x()

.. code:: ipython3

    sc.pl.spatial(adata, color=['cell_type'], size=1.4, alpha=0.7,
                 palette=hn.brca_default_color_celltype)



.. code:: ipython3

    hn.pl.plot_cell_type_proportion(adata, plot_cell_type='stroma')



.. code:: ipython3

    LR_df = hn.pp.load_lr_df()
    expressed_LR_df = hn.pp.get_expressed_lr_df(LR_df, adata, expressed_proportion=0.3)
    expressed_LR_df.head(3)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Ligand_gene_symbol</th>
          <th>Receptor_gene_symbol</th>
          <th>Ligand_location</th>
          <th>LR_Pair</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>A2M</td>
          <td>LRP1</td>
          <td>secreted</td>
          <td>A2M:LRP1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ADAM15</td>
          <td>ITGA5</td>
          <td>plasma membrane</td>
          <td>ADAM15:ITGA5</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ADAM15</td>
          <td>ITGAV</td>
          <td>plasma membrane</td>
          <td>ADAM15:ITGAV</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    w_best = hn.tl.default_w_visium(adata)
    hn.pl.select_w(adata, w_best=w_best)



.. code:: ipython3

    CE_tensor = hn.tl.compute_ce_tensor(adata, lr_df=expressed_LR_df, w_best=w_best)
    CE_tensor_filtered = hn.tl.filter_ce_tensor(CE_tensor, adata, 
                                                lr_df=expressed_LR_df, w_best=w_best)


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   tutorial_folder/tutorial_CE
   tutorial_folder/tutorial_FCE
