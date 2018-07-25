import numpy as np
import torch
from torch.nn import MSELoss
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import ChebConv, SplineConv, graclus, max_pool, max_pool_x

import matplotlib.pyplot as plt
from tqdm import tqdm

from DataSet import HDF5DataSet
from community_pooling import *


from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import community

target = 'irmsd'
h5 = 'graph_atomic.hdf5'
node_feature = ['chainID','name','charge']
edge_attr = ['dist']


train_dataset = HDF5DataSet(root='./',database=h5,
                            node_feature=node_feature,edge_attr=edge_attr,
                            target=target)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)

# d = train_dataset.get(0)
# d.pos2D = manifold_embedding(d.pos)
# cluster = community_detection(d.internal_edge_index,d.num_nodes,edge_attr=1./d.edge_attr**2)
# d2 = community_pooling(cluster,d)
#plot_graph(d,cluster,pooled_data=d2)

batches = []
for b in train_loader:
	batches.append(b)
b = batches[0]

cluster = community_detection(b.internal_edge_index,b.num_nodes,edge_attr=1./b.edge_attr**2)
b2 = community_pooling(cluster,b)

