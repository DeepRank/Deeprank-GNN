# GraphProt

Use Graph CNN to rank conformations.

![alt-text](./graphprot.png)

## Installation

You'll probably need to manually install pytorch geometric
  * pytorch_geometric : https://github.com/rusty1s/pytorch_geometric
  
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

## Graph Interaction Network

Using the graph interaction network is rather simple :


```python
from graphprot.NeuralNet import NeuralNet
from graphprot.ginet import GINet

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

## Custom CNN

It is also possible to define new network architecture and to specify the loss and optimizer to be used during the training. 

```python
train_dataset = HDF5DataSet(root='./',database='graph_residue.hdf5')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
d = train_dataset.get(1)


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
NN = NeuralNet(database, CustomNet,
               node_feature=['type', 'polarity', 'bsa',
                             'depth', 'hse', 'ic', 'pssm'],
               edge_feature=['dist'],
               target='irmsd',
               index=range(400),
               batch_size=64,
               percent=[0.8, 0.2])
molde.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.loss = MSELoss()

model.train(nepoch=50)

```
