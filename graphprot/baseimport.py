#%matplotlib inline

import community
import networkx as nx
import torch

from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

from torch_geometric.data import Data
from community_pooling import *

def graph(pos,edge_index,edge_attr,internal_edge_index):

    x = torch.tensor(list(range(len(pos))))

    edge_index = np.vstack((edge_index,np.flip(edge_index,1))).T
    edge_index = torch.tensor(edge_index,dtype=torch.long)

    edge_attr = np.vstack((edge_attr,edge_attr))
    edge_attr = torch.tensor(edge_attr,dtype=torch.float)

    pos = torch.tensor(pos,dtype=torch.float)

    data = Data(x=x, edge_index=edge_index,
                 edge_attr = edge_attr, pos = pos)

    data.internal_edge_index = torch.tensor(internal_edge_index.T)

    data.pos2D = manifold_embedding(data.pos)

    edge_attr = edge_attr=1./data.edge_attr**2
    edge_attr = None
    cluster = community_detection(data.internal_edge_index,data.num_nodes,edge_attr=edge_attr)
    d2 = community_pooling(cluster,data)
    plot_graph(data,cluster,pooled_data=d2)