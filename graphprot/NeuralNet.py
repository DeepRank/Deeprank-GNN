import numpy as np
from tqdm import tqdm
from time import time

# torch import
import torch
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F

# torch_geometric import
from torch_scatter import scatter_mean
from torch_geometric.data import DataLoader
from torch_geometric.nn import max_pool_x

# graphprot import
from .DataSet import HDF5DataSet, DivideDataSet


class NeuralNet(object):

    def __init__(self, database, Net,
                 node_feature=['type', 'polarity', 'bsa'],
                 edge_feature=['dist'], target='irmsd',
                 batch_size=32, percent=[0.8, 0.2], index=None, database_eval = None):

        # dataset
        dataset = HDF5DataSet(root='./', database=database, index=index,
                              node_feature=node_feature, edge_feature=edge_feature,
                              target=target)
        PreCluster(dataset, method='mcl')

        train_dataset, valid_dataset = DivideDataSet(
            dataset, percent=percent)

        if database_eval is not None :
            valid_dataset = HDF5DataSet(root='./', database=database_eval, index=index,
                                        node_feature=node_feature, edge_feature=edge_feature,
                                        target=target)
            print('Independent validation set loaded')
            PreCluster(dataset_eval, method='mcl')     
        
        else: 
            print('No independent validation set loaded')
            
        # dataloader
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False)
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False)

        # get the device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # put the model
        self.model = Net(dataset.get(0).num_features).to(self.device)

        # optimizer/loss/epoch
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01)
        self.loss = MSELoss()

        # parameters
        self.node_feature = node_feature
        self.edge_feature = edge_feature
        self.target = target

    def train(self, nepoch=1, validate=False):

        self.model.train()
        for epoch in range(1, nepoch+1):
            t0 = time()
            loss = self._epoch(epoch)

            if validate:
                _, val_loss = self.eval(self.valid_loader)
                t = time() - t0
                print('Epoch [%04d] : train loss %e | valid loss %e | time %1.2e sec.' % (
                    epoch, loss, val_loss, t))
            else:
                t = time() - t0
                print('Epoch [%04d] : train loss %e | time %1.2e sec.' % (
                    epoch, loss, t))

    def eval(self, loader):

        self.model.eval()

        loss_func, loss_val = self.loss, 0
        out = []
        for data in loader:
            data = data.to(self.device)
            pred = self.model(data).reshape(-1)
            loss_val += loss_func(pred, data.y)
            out += pred.reshape(-1).tolist()
        return out, loss_val

    def _epoch(self, epoch):

        running_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data).reshape(-1)
            loss = self.loss(out, data.y)
            running_loss += loss.data.item()
            loss.backward()
            self.optimizer.step()
        return running_loss

    def plot_scatter(self):

        import matplotlib.pyplot as plt

        self.model.eval()

        pred, truth = {'train': [], 'valid': []}, {
            'train': [], 'valid': []}

        for data in self.train_loader:
            data = data.to(self.device)
            truth['train'] += data.y.tolist()
            pred['train'] += self.model(data).reshape(-1).tolist()

        for data in self.valid_loader:
            data = data.to(self.device)
            truth['valid'] += data.y.tolist()
            pred['valid'] += self.model(data).reshape(-1).tolist()

        plt.scatter(truth['train'], pred['train'], c='blue')
        plt.scatter(truth['valid'], pred['valid'], c='red')
        plt.show()

    def save_model(self, filename='model.pth.tar'):

        state = {'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'node': self.node_feature,
                 'edge': self.edge_feature,
                 'target': self.target}

        torch.save(state, filename)

    def load_model(self, filename):

        state = torch.load(filename)

        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.node_feature = state['node_feature']
        self.edge_feature = state[edge_feature]
        self.target = target
