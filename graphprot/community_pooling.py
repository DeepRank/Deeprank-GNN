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



def plot_graph(graph,cluster):

    import matplotlib.pyplot as plt
    pos = nx.spring_layout(graph,iterations=200)
    nx.draw(graph,pos,node_color=cluster)
    plt.show()


def community_detection_per_batch(edge_index,batch,num_nodes,edge_attr=None):

    _plot_ = False

    # make the networkX graph
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))

    for iedge,(i,j) in enumerate(edge_index.transpose(0,1).tolist()):
        if edge_attr is None:
            g.add_edge(i,j)
        else:
            g.add_edge(i,j,weight=edge_attr[iedge])

    num_batch = max(batch)+1
    all_index = range(num_nodes)
    cluster, ncluster = [], 0

    for iB in range(num_batch):

        index = torch.tensor(all_index)[batch==iB].tolist()
        subg = g.subgraph(index)

        # detect the communities
        c = community.best_partition(subg)
        cluster += [v+ncluster for k,v in c.items()]
        ncluster = max(cluster)

        if _plot_:
            plot_graph(g,c)

    # return
    device = edge_index.device
    return torch.tensor(cluster).to(device)


def community_detection(edge_index,num_nodes,edge_attr=None,batches=None):

    _plot_ = False

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

    if _plot_:
        if batches is None:
            col = [v for k,v in cluster.items()]
        else:
            col = batches
        plot_graph(g,col)

    # return
    device = edge_index.device
    return torch.tensor([v for k,v in cluster.items()]).to(device)


def community_pooling(cluster,data):


    # determine what the batches has as attributes
    has_internal_edges = hasattr(data,'internal_edge_index')
    has_pos2D = hasattr(data,'pos2D')
    has_pos = hasattr(data,'pos')

    cluster, perm = consecutive_cluster(cluster)
    cluster = cluster.to(data.x.device)

    # pool the node infos
    x, _ = scatter_max(data.x,cluster,dim=0)

    # pool the edges
    edge_index, edge_attr  = pool_edge(cluster,data.edge_index, data.edge_attr)

    # pool internal edges if necessary
    if has_internal_edges:
        internal_edge_index, internal_edge_attr = pool_edge(cluster,data.internal_edge_index, data.internal_edge_attr)

    # pool the pos
    if has_pos:
        pos = scatter_mean(data.pos, cluster, dim=0)
    if has_pos2D:
        pos2D = scatter_mean(data.pos2D, cluster, dim=0)

    # pool batch
    if hasattr(data,'batch'):
        batch = None if data.batch is None else pool_batch(perm, data.batch)
        data = Batch(batch=batch, x=x, edge_index=edge_index,
                     edge_attr = edge_attr, pos = pos)

        if has_internal_edges:
            data.internal_edge_index = internal_edge_index
            data.internal_edge_attr = internal_edge_attr

    else:
        data = Data(x=x, edge_index=edge_index,
                     edge_attr = edge_attr, pos = pos)

        if has_internal_edges:
            data.internal_edge_index = internal_edge_index
            data.internal_edge_attr = internal_edge_attr
        if has_pos2D:
            data.pos2D = pos2D

    return data


if __name__ == "__main__":

    import torch
    from torch_geometric.data import Data, Batch

    edge_index = torch.tensor([[0,1,1,2,3,4,4,5],
                               [1,0,2,1,4,3,5,4]],dtype=torch.long)

    x = torch.tensor([[0],[1],[2],[3],[4],[5]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data.pos = torch.tensor(np.random.rand(data.num_nodes,3))
    c = community_detection(data.edge_index,data.num_nodes)

    batch = Batch().from_data_list([data,data])
    cluster = community_detection(batch.edge_index,batch.num_nodes)
    new_batch = community_pooling(cluster,batch)