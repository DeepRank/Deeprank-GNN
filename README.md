# GraphProt


Use Graph CNN to rank conformations.

## Installation

The code is build so far on all the otjher tools developped in the DeepRank project. Hence you will need:
  . deeprank: https://github.com/DeepRank/deeprank
  . iScore : https://github.com/DeepRank/iScore

You'll also need:
  . pytorch_geometric : https://github.com/rusty1s/pytorch_geometric
  . networkx : https://github.com/networkx/networkx

When all the dependencies are installed just clone the repo and install it with:

```
pip install -e ./
```

## Graphs

Two types of graphs can be generated and use in a siamese network later: Residue graphs and Atomic graphs (the names are self explanatory). These graphs contains the all the node/edge info as:

  . `Graph.node` : the signal on each node. For `Atomic` graphs these are typically the atom name, charge, .... For `Residue` graph these are residue names, BSA, PSSM, ...
  . `Graph.edge_index` : the index of the edges
  . `Grapg.edge_attr`: Attribute of the edges (length, coulomb interaction, vdwaals interaction ...)


To see how the graph look like you can use:

```
python -i AtomicGraph.py
```

or

```
python -i ResidueGraph.py
```

## Line Graphs

Since we are particualrly interested in the properties of the edges I though it would be interesting to study the line graphs (https://en.wikipedia.org/wiki/Line_graph) since the convolution in graph CNN is done on the nodes and not the edges. We can easily generates line graphs with `networkx` as shown in the two scipts mentionned above

## Generate Graphs

All the graphs/line graphs of all the pdb/pssm stored in `data/pdb/` and `data/pssm/` with the `GenGraph.py` script.

```
python GenGraph.py
```

This will generate two hdf5 files `graph_residue.hdf5` and 'linegraph_residue.hdf5' that contains the graph of the different conformations.


## Graph CNN

We can then use do some deeplearning on the graphs using the `HDF5DataSet` class in `DataSet.py` and the model defined in 'model.py':

```
python model.py
```


