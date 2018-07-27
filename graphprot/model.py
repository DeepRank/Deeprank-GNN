import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import GCNConv, ChebConv, SplineConv, NNConv, GATConv
from torch_geometric.nn import graclus, max_pool, max_pool_x
import matplotlib.pyplot as plt
from tqdm import tqdm

from DataSet import HDF5DataSet
from community_pooling import *

from wgat_conv import WGATConv


index = np.arange(400)
#np.random.shuffle(index)

index_train = index[0:50]
index_test = index[0:

50]
batch_size = 10

target = 'irmsd'

# h5 = 'graph_residue.hdf5'
# node_feature = ['type','bsa']
# edge_attr = ['dist','polarity']

h5train = '1ATN_atomic.hdf5'
h5test = '1AK4_atomic.hdf5'

node_feature = ['name','charge','sig']
edge_attr = ['dist']

# h5 = 'linegraph_atomic.hdf5'
# node_feature = ['dist','coulomb','vanderwaals','name_1','name_2','charge_1','charge_2']
# edge_attr = ['name','charge']

train_dataset = HDF5DataSet(root='./',database=h5train,index=index_train,
                            node_feature=node_feature,edge_attr=edge_attr,
                            target=target)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
d = train_dataset.get(0)

test_dataset = HDF5DataSet(root='./',database=h5test,index=index_test,
                           node_feature=node_feature,edge_attr=edge_attr,
                           target=target)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.conv1 = GCNConv(d.num_features, 16)
        # self.conv2 = GCNConv(16, 32)

        #self.conv1 = SplineConv(d.num_features, 16, dim=3, kernel_size=3)
        #self.conv2 = SplineConv(16, 64, dim=3, kernel_size=3)

        # self.conv1 = ChebConv(d.num_features,8,K=3)
        # self.conv2 = ChebConv(8,16,K=3)

        # n1 = nn.Sequential(nn.Linear(2, 25), nn.ReLU(), nn.Linear(25, 32))
        # self.conv1 = NNConv(d.num_features, 32, n1)

        # n2 = nn.Sequential(nn.Linear(2, 25), nn.ReLU(), nn.Linear(25, 2048))
        # self.conv2 = NNConv(32, 64, n2)

        self.conv1 = WGATConv(d.num_features, 16)
        self.conv2 = WGATConv(16 , 32)
        #self.conv3 = WGATConv(32 , 32)

        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data):

        act = nn.Tanhshrink()
        act = F.elu
        act = nn.LeakyReLU(0.25)

        data.x = act(self.conv1(data.x, data.edge_index,data.edge_attr))
        cluster = community_detection(data.internal_edge_index,data.num_nodes,edge_attr=None)
        data = community_pooling(cluster, data)

        data.x = act(self.conv2(data.x, data.edge_index,data.edge_attr))
        cluster = community_detection(data.internal_edge_index,data.num_nodes,edge_attr=None)
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = scatter_mean(x, batch, dim=0)
        x = act(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        return F.elu(self.fc2(x))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss = MSELoss()

def train(epoch):
    model.train()

    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01

    if epoch == 75:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    for data in (train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).reshape(-1)
        loss(out, data.y).backward()
        optimizer.step()


def test():
    model.eval()
    loss_func = MSELoss()
    loss_val = 0

    for data in test_loader:
        data = data.to(device)
        pred = model(data).reshape(-1)
        loss_val += loss_func(pred,data.y)
    return loss_val


for epoch in range(1, 100):
    train(epoch)
    #test_acc = test()
    #print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
    print('Epoch: {:02d}'.format(epoch))

pred, truth = [], []
for data in train_loader:
    truth += data.y.tolist()
    pred += model(data).reshape(-1).tolist()

test_pred, test_truth = [], []
for data in test_loader:
    test_truth += data.y.tolist()
    test_pred += model(data).reshape(-1).tolist()
plt.scatter(truth,pred,c='blue')
plt.scatter(test_truth,test_pred,c='red')
#plt.plot([0,1],[0,1])
plt.show()