API
===

Import infercnvpy together with scanpy as

.. code-block:: python

   import scanpy as sc
   import HoloNet as hn

For consistency, the infercnvpy API tries to follow the `scanpy API <https://scanpy.readthedocs.io/en/stable/api/index.html>`__
as closely as possible.

.. _api-io:

Preprocessing: `pp`
------------------

.. module:: HoloNet.pp

.. autosummary::
   :toctree: ./generated

   get_expressed_lr_df
   read_visium
   
