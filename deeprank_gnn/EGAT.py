import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn


class EGAT(torch.nn.Module):
    """
    1) Create edges feature by concatenating node feature
    :math: `e_{ij} = LeakyReLu (a_{ij} * [W * x_i || W * x_j])`
    
    2) Apply softmax function, in order to learn to ignore some edges
    :math: `\alpha_{ij}  = softmax(e_{ij})`
    
    3) Sum over the nodes (no averaging here !)
    :math: `z_i = \sum_j (\alpha_{ij} * Wx_j + b_i)`
    
    Herein, we add the edge feature to the step 1)
    :math: `e_{ij} = LeakyReLu (a_{ij} * [W * x_i || W * x_j || We * edge_{attr} ])`
    """

    def __init__(self,
                 in_channels,
                 out_channels, 
                 number_edge_features=1,
                 bias=False):

        super(EGAT, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc = nn.Linear(self.in_channels, self.out_channels, bias=bias)
        self.fc_edge_attr = nn.Linear(number_edge_features, 3, bias=bias)
        self.fc_attention = nn.Linear(2 * self.out_channels + 3, 1, bias=bias)
        self.initialize_parameters()
        
    def initialize_parameters(self):
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
            num_node, self.out_channels).to(h.device)
        z = scatter_sum(h, row, dim=0, out=out)
        
        return z
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.in_channels,
                                   self.out_channels)
