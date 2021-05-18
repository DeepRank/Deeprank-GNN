Graph Network
=====================================

EGAT
--------------------------------------

1) Create edges feature by concatenating node feature
.. math::
    `e_{ij} = LeakyReLu (a_{ij} * [W * x_i || W * x_j])`
    
2) Apply softmax function, in order to learn to ignore some edges
.. math::
    `\alpha_{ij}  = softmax(e_{ij})`
    
3) Sum over the nodes (no averaging here !)
.. math::
    `z_i = \sum_j (\alpha_{ij} * Wx_j + b_i)`
    
Herein, we add the edge feature to the step 1)
.. math::
    `e_{ij} = LeakyReLu (a_{ij} * [W * x_i || W * x_j || We * edge_{attr} ])`

.. automodule:: deeprank_gnn.EGAT
    :members:
    :undoc-members:


Fout Net
---------------------------------------


.. automodule:: deeprank_gnn.foutnet
    :members:
    :undoc-members:

GInet
---------------------------------------

.. automodule:: deeprank_gnn.ginet
    :members:
    :undoc-members:

WGAT Conv
---------------------------------------

.. automodule:: deeprank_gnn.wgat_conv
    :members:
    :undoc-members:
