API
===

Import HoloNet as

.. code-block:: python

   import HoloNet as hn


.. _api-io:

Preprocessing: `pp`
-------------------------

Import spatial transcriptomic data into :class:`~anndata.AnnData`.
Extract and filter the ligand-receptor pair dataframe


.. module:: HoloNet.preprocessing

.. autosummary::
   :toctree: ./generated
   
   load_brca_visium_10x
   load_lr_df
   get_expressed_lr_df
   
   
   
Tools: `tl`
------------------

Some tool functions.


.. module:: HoloNet.tools


Constructing multi-view CE network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

.. autosummary::
   :toctree: ./generated

   elements_expr_df_calculate
   compute_ce_tensor
   filter_ce_tensor
   

Computing centralities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

.. autosummary::
   :toctree: ./generated
   
   compute_ce_network_eigenvector_centrality
   compute_ce_network_degree_centrality
   
   
Clustering lr pairs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

.. autosummary::
   :toctree: ./generated
   
   cluster_lr_based_on_ce


Selecting parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

.. autosummary::
   :toctree: ./generated
   
   default_w_visium

   
Predicting: `pr`
------------------

.. module:: HoloNet.predicting


Preparing the inputs of the graph neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

.. autosummary::
   :toctree: ./generated

   get_continuous_cell_type_tensor
   get_one_hot_cell_type_tensor
   get_gene_expr
   get_one_case_expr
   adj_normalize
   
   
Training the graph neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

.. autosummary::
   :toctree: ./generated

   mgc_repeat_training
   get_mgc_result
   mgc_training_for_multiple_targets
   get_mgc_result_for_multiple_targets
   
   
Model saving and loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

.. autosummary::
   :toctree: ./generated   
   
   save_model_list
   load_model_list
   


Plotting: `pl`
------------------

.. module:: HoloNet.plotting

Base plotting methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

.. autosummary::
   :toctree: ./generated
   
   feature_plot
   cell_type_level_network


Plots for spatial datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

.. autosummary::
   :toctree: ./generated
   
   plot_cell_type_proportion


Plotting CEs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

.. autosummary::
   :toctree: ./generated

   ce_hotspot_plot
   ce_cell_type_network_plot
   lr_umap
   lr_clustering_dendrogram
   lr_cluster_ce_hotspot_plot
 
 
Plotting FCEs by interpreting the graph neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

.. autosummary::
   :toctree: ./generated
   
   lr_rank_in_mgc
   fce_cell_type_network_plot
   delta_e_proportion
   plot_mgc_result
   
   
Plots for identifying genes dominated by cell–cell communication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

.. autosummary::
   :toctree: ./generated
   
   find_genes_linked_to_ce
   detect_pathway_related_genes
   
   
Plots for selecting parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     

.. autosummary::
   :toctree: ./generated
   
   select_w
   
   
