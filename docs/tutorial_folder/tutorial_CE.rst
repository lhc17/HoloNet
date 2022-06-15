CE visualization
==================

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



.. image:: tutorial_CE_files/tutorial_CE_2_0.png


.. code:: ipython3

    hn.pl.plot_cell_type_proportion(adata, plot_cell_type='stroma')



.. image:: tutorial_CE_files/tutorial_CE_3_0.png


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



.. image:: tutorial_CE_files/tutorial_CE_5_0.png


.. code:: ipython3

    CE_tensor = hn.tl.compute_ce_tensor(adata, lr_df=expressed_LR_df, w_best=w_best)
    CE_tensor_filtered = hn.tl.filter_ce_tensor(CE_tensor, adata, 
                                                lr_df=expressed_LR_df, w_best=w_best)


.. parsed-literal::

    100%|██████████| 286/286 [35:28<00:00,  7.44s/it]


.. code:: ipython3

    cell_type_mat, \
    cell_type_names = hn.pr.get_continuous_cell_type_tensor(adata, continuous_cell_type_slot = 'predicted_cell_type',)

.. code:: ipython3

    hn.pl.ce_hotspot_plot(CE_tensor_filtered, adata, 
                          lr_df=expressed_LR_df, plot_lr='COL1A1:DDR1')



.. image:: tutorial_CE_files/tutorial_CE_8_0.png


.. code:: ipython3

    hn.pl.ce_hotspot_plot(CE_tensor_filtered, adata, 
                          lr_df=expressed_LR_df, plot_lr='COL1A1:DDR1',
                          centrality_measure='eigenvector')



.. image:: tutorial_CE_files/tutorial_CE_9_0.png


.. code:: ipython3

    _ = hn.pl.ce_cell_type_network_plot(CE_tensor_filtered, cell_type_mat, cell_type_names,
                                        lr_df=expressed_LR_df, plot_lr='COL1A1:DDR1', edge_thres=0.2,
                                        palette=hn.brca_default_color_celltype)



.. image:: tutorial_CE_files/tutorial_CE_10_0.png


.. code:: ipython3

    cell_cci_centrality = hn.tl.compute_ce_network_eigenvector_centrality(CE_tensor_filtered)
    clustered_expressed_LR_df = hn.tl.cluster_lr_based_on_ce(CE_tensor_filtered, adata, expressed_LR_df, 
                                                             w_best=w_best, cell_cci_centrality=cell_cci_centrality)

.. code:: ipython3

    hn.pl.lr_umap(clustered_expressed_LR_df, cell_cci_centrality, plot_lr_list=['COL1A1:DDR1'], linewidths=0.7)



.. image:: tutorial_CE_files/tutorial_CE_12_0.png


.. code:: ipython3

    hn.pl.lr_cluster_ce_hotspot_plot(lr_df=clustered_expressed_LR_df,
                                     cell_cci_centrality=cell_cci_centrality,
                                     adata=adata)



.. image:: tutorial_CE_files/tutorial_CE_13_0.png



.. image:: tutorial_CE_files/tutorial_CE_13_1.png



.. image:: tutorial_CE_files/tutorial_CE_13_2.png



.. image:: tutorial_CE_files/tutorial_CE_13_3.png

