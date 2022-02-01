Graph Neural Networks
=====================================

GINet layer
--------------------------------------

Graph Interaction Networks layer

This layer is inspired by Sazan Mahbub et al. "EGAT: Edge Aggregated Graph Attention Networks and Transfer Learning Improve Protein-Protein Interaction Site Prediction", BioRxiv 2020

1) Create edges feature by concatenating node feature

.. math::
    e_{ij} = LeakyReLu (a_{ij} * [W * x_i || W * x_j])
    
2) Apply softmax function, in order to learn to consider or ignore some neighboring nodes

.. math::
    \alpha_{ij}  = softmax(e_{ij})
    
3) Sum over the nodes (no averaging here)

.. math::
    z_i = \sum_j (\alpha_{ij} * Wx_j + b_i)
    
Herein, we add the edge feature to the step 1)

.. math::
    e_{ij} = LeakyReLu (a_{ij} * [W * x_i || W * x_j || We * edge_{attr} ])


.. automodule:: deeprank_gnn.ginet
    :members:
    :undoc-members:


Fout Net layer
---------------------------------------

This layer is described by eq. (1) of "Protein Interface Predition using Graph Convolutional Network", by Alex Fout et al. NIPS 2018

.. math::
    z = x_i * Wc + 1 / Ni Sum_j x_j * Wn + b

.. automodule:: deeprank_gnn.foutnet
    :members:
    :undoc-members:

sGraphAttention (sGAT) layer
---------------------------------------

This is a new layer that is similar to the graph attention network but simpler

.. math::   
    z_i = 1 / Ni Sum_j a_ij * [x_i || x_j] * W + b_i

|| is the concatenation operator: [1,2,3] || [4,5,6] = [1,2,3,4,5,6] Ni is the number of neighbor of node i Sum_j runs over the neighbors of node i :math:`a_ij` is the edge attribute between node i and j

.. automodule:: deeprank_gnn.sGat
    :members:
    :undoc-members:

