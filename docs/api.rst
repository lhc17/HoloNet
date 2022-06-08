API
===

Import HoloNet as

.. code-block:: python

   import HoloNet as hn


.. _api-io:

Preprocessing: `pp`
------------------

Import spatial transcriptomic data into :class:`~anndata.AnnData`.
Extract and filter the ligand-receptor pair dataframe


.. module:: HoloNet.preprocessing

.. autosummary::
   :toctree: ./generated

   get_expressed_lr_df
   read_visium
   
   
   
Tools: `tl`
------------------

Some tool functions.


.. module:: HoloNet.tools


Constructing multi-view CE network
^^^^^^^^

.. autosummary::
   :toctree: ./generated

   compute_ce_tensor
   filter_ce_tensor
   

Computing centralities
^^^^^^^^

.. autosummary::
   :toctree: ./generated
   
   compute_ce_network_eigenvector_centrality
   compute_ce_network_degree_centrality
   
   
Clustering lr pairs
^^^^^^^^

.. autosummary::
   :toctree: ./generated
   
   cluster_lr_based_on_ce


Other functions
^^^^^^^^

.. autosummary::
   :toctree: ./generated
   
   dist_factor_calculate
   default_w_visium

   
Predicting: `pr`
------------------

.. module:: HoloNet.predicting

.. autosummary::
   :toctree: ./generated

   mgc_repeat_training
   get_mgc_result
   mgc_training_with_single_view
   mgc_training_for_multiple_targets
   get_mgc_result_for_multiple_targets
   adj_normalize
   train_test_mask
   get_continuous_cell_type_tensor
   get_one_hot_cell_type_tensor
   save_model_list
   load_model_list
   get_gene_expr
   get_one_case_expr


Plotting: `pl`
------------------

.. module:: HoloNet.plotting

.. autosummary::
   :toctree: ./generated

   ce_hotspot_plot
   ce_cell_type_network_plot
   lr_rank_in_mgc
   fce_cell_type_network_plot
   delta_e_proportion
   save_mgc_interpretation_for_all_target
   plot_mgc_result
   find_genes_linked_to_ce
   single_view_mgc_coef_plot
   feature_plot
   cell_type_level_network
   plot_cell_type_proportion
   select_w
   lr_cluster_ce_hotspot_plot
   lr_umap
  
 
 
