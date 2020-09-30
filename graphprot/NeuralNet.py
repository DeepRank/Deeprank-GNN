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
                 batch_size=32, percent=[0.8, 0.2], index=None, database_eval = None,
                 class_weights = None, task = 'class', classes = [0,1]):

        # dataset
        dataset = HDF5DataSet(root='./', database=database, index=index,
                              node_feature=node_feature, edge_feature=edge_feature,
                              target=target)
        PreCluster(dataset, method='mcl')

        train_dataset, valid_dataset = DivideDataSet(
            dataset, percent=percent)

        # independent validation dataset
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

        # parameters
        self.node_feature = node_feature
        self.edge_feature = edge_feature
        self.target = target
        self.task = task
        self.class_weights = class_weights

        # put the model
        if self.task == 'reg' :
            self.model = Net(dataset.get(0).num_features).to(self.device)
            
        elif self.task == 'class' :
            self.classes = classes
            self.classes_idx = {i: idx for idx, i in enumerate(self.classes)}
            self.output_shape = len(self.classes)
            try :
                self.model = Net(dataset.get(0).num_features, self.output_shape).to(self.device)
            except :
                raise ValueError(
                    f"The loaded model does not accept output_shape = {self.output_shape} argument \n\t"
                    f"Check your input or adapt the model\n\t"
                    f"Example :\n\t"
                    f"def __init__(self, input_shape): --> def __init__(self, input_shape, output_shape) \n\t"
                    f"self.fc2 = torch.nn.Linear(64, 1) --> self.fc2 = torch.nn.Linear(64, output_shape) \n\t")
        
        # optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01)
        
        # loss 
        if self.task == 'reg':
            self.loss = MSELoss()

        elif self.task == 'class':
            self.loss = nn.CrossEntropyLoss(weight = self.class_weights, reduction='mean')

        else:
            raise ValueError(
                f"Task {self.task} not recognized. Options are:\n\t "
                f"reg': regression \n\t 'class': classifiation\n")        


    def train(self, nepoch=1, validate=False):

        self.model.train()
        for epoch in range(1, nepoch+1):
            t0 = time()
            acc, loss = self._epoch(epoch)

            if validate:
                _, val_acc, val_loss = self.eval(self.valid_loader)
                t = time() - t0
                if acc is not None :
                    print('Epoch [%04d] : train loss %e | train accuracy %1.4e | valid loss %e | val accuracy %1.4e | time %1.2e sec.' % (
                        epoch, loss, acc, val_loss, val_acc, t))
                else :
                    print('Epoch [%04d] : train loss %e | valid loss %e | time %1.2e sec.' % (
                        epoch, loss, val_loss, t))

            else:
                t = time() - t0
                if acc is not None :
                    print('Epoch [%04d] : train loss %e | accuracy %1.4e | time %1.2e sec. |' % (
                    epoch, loss, acc, t))
                else :
                    print('Epoch [%04d] : train loss %e | time %1.2e sec. |' % (
                    epoch, loss, t))


    def calcAccuracy(self, prediction, target, reduce=True):

        overlap = (prediction.argmax(dim=1)==target).sum()
        if reduce:
            return overlap/float(target.size()[-1])

        return overlap


    def eval(self, loader):

        with self.model.eval():

            loss_func, loss_val = self.loss, 0
            out = []
            acc = []
            for data in loader:
                data = data.to(self.device)
                pred = self.model(data)

                if self.task == 'class':
                    pred = F.softmax(pred, dim=1)
                    data.y = torch.tensor([self.classes_idx[int(x)] for x in data.y])
                    acc.append(self.calcAccuracy(pred, data.y))

                else :
                    out = out.reshape(-1)
                    acc = None

                loss_val += loss_func(pred, data.y)
                out += pred.reshape(-1).tolist()

        if self.task == 'class':
            return out, torch.mean(torch.stack(acc)), loss_val

        else :
            return out, acc, loss_val
            
            
    def _epoch(self, epoch):

        running_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)

            if self.task == 'class':
                out = F.softmax(out, dim=1)
                data.y = torch.tensor([self.classes_idx[int(x)] for x in data.y])
                acc.append(self.calcAccuracy(out, data.y))

            else :
                out = out.reshape(-1)
                acc = None

            loss = self.loss(out, data.y)
            running_loss += loss.data.item()
            loss.backward()
            self.optimizer.step()

        if self.task == 'class':
            return torch.mean(torch.stack(acc)), running_loss

        else :
            return acc, running_loss


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
        self.node_feature = state['node']
        self.edge_feature = state['edge']
        self.target = state['target']
