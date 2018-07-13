# GraphProt


Use Graph CNN to rank conformations.

## Installation

The code is build so far on all the otjher tools developped in the DeepRank project. Hence you will need:
  * deeprank: https://github.com/DeepRank/deeprank
  * iScore : https://github.com/DeepRank/iScore

You'll also need:
  * pytorch_geometric : https://github.com/rusty1s/pytorch_geometric
  * networkx : https://github.com/networkx/networkx

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

Since we are particualrly interested in the properties of the edges I though it would be interesting to study the line graphs since the convolution in graph CNN is done on the nodes and not the edges. We can easily generates line graphs with `networkx` as shown in the two scipts mentionned above


```python
from Graph import Graph
from ResidueGraph import ResidueGraph

pdb = 'data/pdb/1ATN.pdb'
pssm = {'A':'./data/pssm/1ATN.A.pdb.pssm','B':'./data/pssm/1ATN.B.pdb.pssm'}
graph = ResidueGraph(pdb,pssm)
line_graph = Graph.get_line_graph(graph)
```

## Generate Graphs

All the graphs/line graphs of all the pdb/pssm stored in `data/pdb/` and `data/pssm/` with the `GenGraph.py` script. This will generate two hdf5 files `graph_residue.hdf5` and 'linegraph_residue.hdf5' that contains the graph of the different conformations.


```python
from GraphGen import GraphHDF5

pdb_path = './data/pdb'
pssm_path = './data/pssm'
ref = './data/ref'

GraphHDF5(pdb_path=pdb_path,ref_path=ref,pssm_path=pssm_path,
	      graph_type='residue',outfile='graph_residue.hdf5')
```

## Graph CNN

We can then use do some deeplearning on the graphs using the `HDF5DataSet` class in `DataSet.py` and the model defined in 'model.py':

```python

import torch
from torch.nn import MSELoss
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import SplineConv, graclus, max_pool, max_pool_x

from DataSet import HDF5DataSet

train_dataset = HDF5DataSet(root='./',database='graph_residue.hdf5')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
d = train_dataset.get(1)


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class Net(torch.nn.Module):
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
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss = MSELoss()

def train(epoch):
    model.train()

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 26:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss(model(data)[0], data.y).backward()
        optimizer.step()

for epoch in range(50):
     train(epoch)
     print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))

```
