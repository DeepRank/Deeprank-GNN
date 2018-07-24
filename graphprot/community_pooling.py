import community
import networkx as nx
import torch

from torch_scatter import scatter_max
from torch_geometric.nn.pool.pool import pool_edge, pool_batch, pool_pos
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.data import Batch, Data


from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_graph(data,cluster):

    color = cluster.tolist()

    n_components = 2
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)

    Y = tsne.fit_transform(data.pos)
    plt.scatter(Y[:,0],Y[:,1],c=color,cmap=plt.cm.tab10)

    for i in range(len(Y)):
        plt.text(Y[i,0],Y[i,1],str(color[i]))

    for i,j in data.internal_edge_index.transpose(0,1).tolist():
        plt.plot([Y[i,0],Y[j,0]],[Y[i,1],Y[j,1]],c='black')
    plt.show()



def community_detection(edge_index,num_nodes):

    # make the networkX graph
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    for i,j in edge_index.transpose(0,1).tolist():
        g.add_edge(i,j)

    # detect the communities
    cluster = community.best_partition(g)

    # return
    return torch.tensor([v for k,v in cluster.items()])

def community_pooling(cluster,data):

    cluster,perm = consecutive_cluster(cluster)

    # pool the node infos
    x, _ = scatter_max(data.x,cluster,dim=0)

    # pool the edges
    edge_index, edge_attr  = pool_edge(cluster,data.edge_index, data.edge_attr)
    internal_edge_index, _ = pool_edge(cluster,data.internal_edge_index, None)

    # pool the pos
    pos = pool_pos(data.pos,cluster)

    # pool batch
    if hasattr(data,batch):
        batch = None if data.batch is None else pool_batch(perm, data.batch)
        return Batch(batch=batch, x=x, edge_index=index,
                     edge_attr = attr, pos = pos,
                     internal_edge_index = internal_edge_index)

    else:
        return Data(x=x, edge_index=index,
                     edge_attr = attr, pos = pos,
                     internal_edge_index = internal_edge_index)