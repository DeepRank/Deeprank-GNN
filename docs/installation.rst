Installation
=========================

Via Python Package
-----------------------------

The latest release of DeepRank-GNN can be installed using the pypi package manager with :

``pip install deeprank-gnn``

If you are planning to only use DeepRank-GNN this is the quickest way to obtain the code


Via GitHub
-------------

For user who would like to contribute, the code is hosted on GitHub_ (https://github.com/DeepRank/DeepRank-GNN)

.. _GitHub: https://github.com/DeepRank/DeepRank-GNN

To install the code

 * clone the repository ``git clone https://github.com/DeepRank/DeepRank-GNN.git``
 * go there ``cd DeepRank-GNN``
 * install the module ``pip install -e ./``

You can then test the installation :

 * ``cd test``
 * ``pytest``

.. note::
  Ensure that at least PyTorch 1.5.0 is installed:
  
  ``python -c "import torch; print(torch.__version__)``
  
  In case PyTorch Geometric installation fails, refer to the installation guide:  https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html 


