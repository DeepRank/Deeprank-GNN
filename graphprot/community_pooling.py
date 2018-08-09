import community
import networkx as nx
import torch

from torch_scatter import scatter_max, scatter_mean
from torch_geometric.nn.pool.pool import pool_edge, pool_batch, pool_pos
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.data import Batch, Data


from sklearn import manifold, datasets
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def community_detection(edge_index,num_nodes,edge_attr=None):

    # make the networkX graph
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    for iedge,(i,j) in enumerate(edge_index.transpose(0,1).tolist()):
        if edge_attr is None:
            g.add_edge(i,j)
        else:
            g.add_edge(i,j,weight=edge_attr[iedge])

    # detect the communities
    cluster = community.best_partition(g)

    # return
    device = edge_index.device
    return torch.tensor([v for k,v in cluster.items()]).to(device)

def community_pooling(cluster,data):

    cluster,perm = consecutive_cluster(cluster)
    cluster = cluster.to(data.x.device)

    # pool the node infos
    x, _ = scatter_max(data.x,cluster,dim=0)

    # pool the edges
    edge_index, edge_attr  = pool_edge(cluster,data.edge_index, data.edge_attr)
    internal_edge_index, internal_edge_attr = pool_edge(cluster,data.internal_edge_index, data.internal_edge_attr)

    # pool the pos
    pos = scatter_mean(data.pos, cluster, dim=0)
    if hasattr(data,'pos2D'):
        pos2D = scatter_mean(data.pos2D, cluster, dim=0)

    # pool batch
    if hasattr(data,'batch'):
        batch = None if data.batch is None else pool_batch(perm, data.batch)
        data = Batch(batch=batch, x=x, edge_index=edge_index,
                     edge_attr = edge_attr, pos = pos)
        data.internal_edge_index = internal_edge_index
        data.internal_edge_attr = internal_edge_attr

    else:
        data = Data(x=x, edge_index=edge_index,
                     edge_attr = edge_attr, pos = pos)
        data.internal_edge_index = internal_edge_index
        data.internal_edge_attr = internal_edge_attr
        if hasattr(data,'pos2D'):
            data.pos2D = pos2D

    return data