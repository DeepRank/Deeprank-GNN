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

def manifold_embedding(pos,method='tsne'):
    n_components = 2
    n_neighbors = 100

    if method == 'tsne':
        tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
        Y = tsne.fit_transform(pos)
    elif method == 'spectral':
        se = manifold.SpectralEmbedding(n_components=n_components,n_neighbors=n_neighbors)
        Y = se.fit_transform(pos)
    elif method == 'mds':
        mds = manifold.MDS(n_components, max_iter=100, n_init=1)
        Y = mds.fit_transform(pos)
    return torch.tensor(Y)

def plot_graph(data,cluster,point_size=None,edge_color='grey'):


    # if pooled_data is not None:

    #     for i,j in pooled_data.internal_edge_index.transpose(0,1).tolist():
    #         plt.plot([pooled_data.pos2D[i,0],pooled_data.pos2D[j,0]],[pooled_data.pos2D[i,1],pooled_data.pos2D[j,1]],c='black')

    #     plt.scatter(pooled_data.pos2D[:,0],pooled_data.pos2D[:,1],s=1000,c='white',edgecolor='black')

    for i,j in data.internal_edge_index.transpose(0,1).tolist():
            plt.plot([data.pos2D[i,0],data.pos2D[j,0]],[data.pos2D[i,1],data.pos2D[j,1]],c=edge_color,alpha=0.4)

    color = cluster.tolist()
    ncluster = len(set(cluster))

    #cmap = plt.cm.tab10
    cmap = colors.ListedColormap(np.random.rand(ncluster,3))

    if point_size is None:
        plt.scatter(data.pos2D[:,0],data.pos2D[:,1],c=color,cmap=cmap)
    else:
        plt.scatter(data.pos2D[:,0],data.pos2D[:,1],c=color,cmap=cmap,s=point_size)
    # for i in range(len(data.pos2D)):
    #     txt = str(color[i]) + {0:'_A',1:'_B'}[int(data.x.tolist()[i][0])]
    #     plt.text(data.pos2D[i,0],data.pos2D[i,1],txt)

    #plt.show()



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
        data.pos2D = pos2D

    return data