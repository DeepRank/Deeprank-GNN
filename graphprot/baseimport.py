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

def graph(pos,edge_index,edge_attr,internal_edge_index,internal_edge_attr,pos2D=None):

    x = torch.tensor(list(range(len(pos))))

    edge_index = np.vstack((edge_index,np.flip(edge_index,1))).T
    edge_index = torch.tensor(edge_index,dtype=torch.long)

    edge_attr = np.vstack((edge_attr,edge_attr))
    edge_attr = torch.tensor(edge_attr,dtype=torch.float)

    pos = torch.tensor(pos,dtype=torch.float)

    data = Data(x=x, edge_index=edge_index,
                 edge_attr = edge_attr, pos = pos)

    data.internal_edge_index = torch.tensor(internal_edge_index.T)
    data.internal_edge_attr = torch.tensor(internal_edge_attr)

    if pos2D is None:
        data.pos2D = manifold_embedding(data.pos)
    else:
        data.pos2D = pos2D

    cluster = community_detection(data.internal_edge_index,data.num_nodes,
                                  edge_attr=data.internal_edge_attr)

    data2 = community_pooling(cluster,data)
    cluster2 = community_detection(data2.internal_edge_index,data2.num_nodes,
                                  edge_attr=data2.internal_edge_attr)

    plot_graph(data2,cluster2,point_size=1000,edge_color='black')
    plot_graph(data,cluster)
    plt.show()

    if pos2D is  None:
        return data.pos2D
    else:
        return None