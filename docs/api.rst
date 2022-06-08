API
===

Import infercnvpy together with scanpy as

.. code-block:: python

   import scanpy as sc
   import HoloNet as hn


.. _api-io:

Preprocessing: `pp`
------------------

.. module:: HoloNet.pp

.. autosummary::
   :toctree: ./generated

   get_expressed_lr_df
   read_visium
   
   
   
Preprocessing: `tl`
------------------

.. module:: HoloNet.tl

.. autosummary::
   :toctree: ./generated

   compute_ce_tensor
   filter_ce_tensor
   compute_ce_network_eigenvector_centrality
   compute_ce_network_degree_centrality
   dist_factor_calculate
   default_w_visium
   cluster_lr_based_on_ce

   
Preprocessing: `pr`
------------------

.. module:: HoloNet.pr

.. autosummary::
   :toctree: ./generated

   MGC_Model
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


Preprocessing: `pl`
------------------

.. module:: HoloNet.pl

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
  
 
 
