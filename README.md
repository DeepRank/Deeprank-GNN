# :warning: Archiving Note

This repository is no longer being maintained and has been archived for historical purposes. 

There is a new version called [DeepRank-GNN-esm](https://github.com/haddocking/DeepRank-GNN-esm) which incorporates the ESM embeddings as replacement of the PSSM profile as features. This version is accessible from a separate repository [here](https://github.com/haddocking/DeepRank-GNN-esm). Next to providing ESM features, it has the same functionalities as the original DeepRank-GNN version. For details refer to the following [publication](https://doi.org/10.1093/bioadv/vbad191).

Next to that, there is a new [DeepRank2](https://github.com/DeepRank/deeprank2) version, an improved and unified version of DeepRank-GNN, [DeepRank](https://github.com/DeepRank/deeprank), and [DeepRank-Mut](https://github.com/DeepRank/DeepRank-Mut).

:sparkles: DeepRank2 allows for transformation and storage of 3D representations of both protein-protein interfaces (PPIs) and protein single-residue variants (SRVs) into either graphs or volumetric grids containing structural and physico-chemical information. These can be used for training neural networks for a variety of patterns of interest, using either our pre-implemented training pipeline for graph neural networks (GNNs) or convolutional neural networks (CNNs) or external pipelines.

- :wrench: **Pull Requests** at [github.com/DeepRank/deeprank2/pulls](https://github.com/DeepRank/deeprank2/pulls)
- :bug: **Bugs**: Reports of bugs can be filed agains our new repo [github.com/DeepRank/deeprank2/issues](https://github.com/DeepRank/deeprank2/issues)
- :star: **Feature Requests**: Add your request or discuss the project w/ the community at [github.com/DeepRank/deeprank2/issues](https://github.com/DeepRank/deeprank2/issues)

We look forward to seeing you in our new space - [DeepRank2](https://github.com/DeepRank/deeprank2)!

# DeepRank-GNN


[![Build Status](https://github.com/DeepRank/DeepRank-GNN/workflows/build/badge.svg)](https://github.com/DeepRank/DeepRank-GNN/actions)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f3f98b2d1883493ead50e3acaa23f2cc)](https://app.codacy.com/gh/DeepRank/DeepRank-GNN?utm_source=github.com&utm_medium=referral&utm_content=DeepRank/DeepRank-GNN&utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/DeepRank/Deeprank-GNN/badge.svg?branch=master)](https://coveralls.io/github/DeepRank/Deeprank-GNN?branch=master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5705564.svg)](https://doi.org/10.5281/zenodo.5705564)

![alt-text](./deeprank_gnn.png)

## Installation

Before installing DeepRank-GNN you need to install pytorch_geometric according to your needs. You can find detailled instructions here :
  * pytorch_geometric : https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

By default the CPU version of pytorch will be installed but you can also customize that installation following the instructions at:
  * pytorch : https://pytorch.org/ 

Once the dependencies installed, you can install the latest release of DeepRank-GNN using the PyPi package manager:

```
pip install DeepRank-GNN
```

Alternatively you can get all the new developments by cloning the repo and installing the code with

```
git clone https://github.com/DeepRank/Deeprank-GNN 
cd DeepRank-GNN
pip install -e ./
```

The documentation can be found here : https://deeprank-gnn.readthedocs.io/ 

## Generate Graphs

All the graphs/line graphs of all the pdb/pssm stored in `data/pdb/` and `data/pssm/` with the `GenGraph.py` script. This will generate the hdf5 file `graph_residue.hdf5` which contains the graph of the different conformations.


```python
from GraphGenMP import GraphHDF5

pdb_path = './data/pdb'
pssm_path = './data/pssm'
ref = './data/ref'

GraphHDF5(pdb_path=pdb_path,ref_path=ref,pssm_path=pssm_path,
	      graph_type='residue',outfile='graph_residue.hdf5')
```

## Graph Interaction Network

Using the graph interaction network is rather simple :


```python
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import GINet

database = './hdf5/1ACB_residue.hdf5'

NN = NeuralNet(database, GINet,
               node_feature=['type', 'polarity', 'bsa',
                             'depth', 'hse', 'ic', 'pssm'],
               edge_feature=['dist'],
               target='irmsd',
               index=range(400),
               batch_size=64,
               percent=[0.8, 0.2])

NN.train(nepoch=250, validate=False)
NN.plot_scatter()
```

## Custom GNN

It is also possible to define new network architecture and to specify the loss and optimizer to be used during the training.

```python


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class CustomNet(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(d.num_features, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight)
        data = max_pool(cluster, data)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight)
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = scatter_mean(x, batch, dim=0)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(database, CustomNet,
               node_feature=['type', 'polarity', 'bsa',
                             'depth', 'hse', 'ic', 'pssm'],
               edge_feature=['dist'],
               target='irmsd',
               index=range(400),
               batch_size=64,
               percent=[0.8, 0.2])
model.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.loss = MSELoss()

model.train(nepoch=50)

```

## h5x support

After installing  `h5xplorer`  (https://github.com/DeepRank/h5xplorer), you can execute the python file `deeprank_gnn/h5x/h5x.py` to explorer the connection graph used by DeepRank-GNN. The context menu (right click on the name of the structure) allows to automatically plot the graphs using `plotly` as shown below.

![alt-text](./h5_deeprank_gnn.png)
