import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import uniform
from torch_scatter import scatter_mean


def add_self_loops_wattr(edge_index, edge_attr, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    dtype, device = edge_index.dtype, edge_index.device
    loop = torch.arange(0, num_nodes, dtype=dtype, device=device)
    loop = loop.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop], dim=1)

    dtype, device = edge_attr.dtype, edge_attr.device
    loop = torch.ones(num_nodes,dtype=dtype, device=device)
    edge_attr = torch.cat([edge_attr,loop])

    return edge_index, edge_attr


class FoutNet(torch.nn.Module):

    """
    This layher is described by eq. (1) of
    Protein Interface Predition using Graph Convolutional Network
    by Alex Fout et al. NIPS 2018

    z =   x_i * Wc + 1 / Ni \Sum_j x_j * Wn + b


    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):

        super(FoutNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Wc and Wn are the center and neighbor weight matrix
        self.Wc = Parameter(torch.Tensor(in_channels, out_channels))
        self.Wn = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.Wc)
        uniform(size, self.Wn)
        uniform(size, self.bias)

    def forward(self, x, edge_index):

        row, col = edge_index
        num_node = len(x)

        # alpha = x * Wc
        alpha = torch.mm(x,self.Wc)

        # beta = x * Wn
        beta = torch.mm(x,self.Wn)

        # gamma_i = 1/Ni Sum_j x_j * Wn
        # there might be a better way than looping over the nodes
        gamma = torch.zeros(num_node,self.out_channels).to(alpha.device)
        for n in range(num_node):
            index = edge_index[:,edge_index[0,:]==n][1,:]
            gamma[n,:] = torch.mean(beta[index,:],dim=0)

        # alpha = alpha + beta
        alpha = alpha + beta

        # add the bias
        if self.bias is not None:
            alpha = alpha + self.bias

        return alpha

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)
