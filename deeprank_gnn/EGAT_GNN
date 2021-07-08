import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn

from torch_scatter import scatter_mean
from torch_scatter import scatter_sum

from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

# torch_geometric import
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import max_pool_x
from torch_geometric.data import DataLoader

# deeprank_gnn import
from deeprank_gnn.community_pooling import get_preloaded_cluster, community_pooling
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.DataSet import HDF5DataSet, PreCluster

class EGAT(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 number_edge_features=1,
                 bias=False):

        super(EGAT, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc = nn.Linear(self.in_channels, self.out_channels, bias=bias)
        self.fc_edge_attr = nn.Linear(number_edge_features, number_edge_features, bias=bias)
        self.fc_attention = nn.Linear(2 * self.out_channels + number_edge_features, 1, bias=bias)
        self.reset_parameters()
        
    def reset_parameters(self):
        
        size = self.in_channels
        uniform(size, self.fc.weight)
        uniform(size, self.fc_attention.weight)
        uniform(size, self.fc_edge_attr.weight)
        
    def forward(self, x, edge_index, edge_attr):

        row, col = edge_index
        num_node = len(x)
        edge_attr = edge_attr.unsqueeze(
            -1) if edge_attr.dim() == 1 else edge_attr

        xcol = self.fc(x[col])
        xrow = self.fc(x[row])
        
        ed = self.fc_edge_attr(edge_attr)
        # create edge feature by concatenating node feature
        alpha = torch.cat([xrow, xcol, ed], dim=1)
        alpha = self.fc_attention(alpha)
        alpha = F.leaky_relu(alpha)
        
        alpha = F.softmax(alpha, dim=1)
        h = alpha * xcol 
        
        out = torch.zeros(
            num_node, self.out_channels).to(alpha.device)
        z = scatter_sum(h, row, dim=0, out=out)

        return z
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.in_channels,
                                   self.out_channels)



class EGAT_Net(torch.nn.Module):
    def __init__(self, input_shape, output_shape = 1):
        super(GIN_EGAT, self).__init__()
        self.conv1 = EGAT(input_shape, 16)
        self.conv2 = EGAT(16, 32)

        self.conv1_ext = EGAT(input_shape, 16)
        self.conv2_ext = EGAT(16, 32)

        self.fc1 = nn.Linear(2*32, 128)
        self.fc2 = nn.Linear(128, output_shape)
        self.clustering = 'mcl'
        self.dropout = 0.4

    def forward(self, data):
        act = F.relu
        data_ext = data.clone()

        # EXTERNAL INTERACTION GRAPH
        # first conv block                                                                                                                                                  
        data.x = act(self.conv1(
            data.x, data.edge_index, data.edge_attr))
        cluster = get_preloaded_cluster(data.cluster0, data.batch)
        data = community_pooling(cluster, data)

        # second conv block                                                                                                                                                    
        data.x = act(self.conv2(
            data.x, data.edge_index, data.edge_attr))
        cluster = get_preloaded_cluster(data.cluster1, data.batch)
        x, batch = max_pool_x(cluster, data.x, data.batch)

        # INTERNAL INTERACTION GRAPH
        # first conv block                                                                                                                                                  
        data_ext.x = act(self.conv1_ext(
            data_ext.x, data_ext.edge_index, data_ext.edge_attr))
        cluster = get_preloaded_cluster(data_ext.cluster0, data_ext.batch)
        data_ext = community_pooling(cluster, data_ext)

        # second conv block                                                                                                                                                    
        data_ext.x = act(self.conv2_ext(
            data_ext.x, data_ext.edge_index, data_ext.edge_attr))
        cluster = get_preloaded_cluster(data_ext.cluster1, data_ext.batch)
        x_ext, batch_ext = max_pool_x(cluster, data_ext.x, data_ext.batch)

        # FC                                                                                                                                                         
        x = scatter_mean(x, batch, dim=0)
        x_ext = scatter_mean(x_ext, batch_ext, dim=0)

        x = torch.cat([x, x_ext], dim=1)
        x = act(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)

        return x
