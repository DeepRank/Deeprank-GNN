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
        
        else:
            raise ValueError(
                f"Task {self.task} not recognized. Options are:\n\t "
                f"reg': regression \n\t 'class': classifiation\n") 
            
        # optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01)
        
        # loss 
        if self.task == 'reg':
            self.loss = MSELoss()

        elif self.task == 'class':
            self.loss = nn.CrossEntropyLoss(weight = self.class_weights, reduction='mean')       


    def train(self, nepoch=1, validate=False, plot=False):

        self.model.train()
        _train_acc = []
        _train_loss = []
        _valid_acc = []
        _valid_loss = []

        for epoch in range(1, nepoch+1):
            t0 = time()
            acc, loss = self._epoch(epoch)
            t = time() - t0
            _train_loss.append(loss)

            if acc is not None:
                _train_acc.append(acc)
                print('Epoch [%04d] : train loss %e | accuracy %1.4e | time %1.2e sec.' % (epoch, loss, acc, t))
            else:
                print('Epoch [%04d] : train loss %e | accuracy None | time %1.2e sec.' % (epoch, loss, t))
                          
            if validate:
                _, val_acc, val_loss = self.eval(self.valid_loader)
                t = time() - t0
                _valid_loss.append(val_loss)

                if acc is not None :
                    _valid_acc.append(val_acc)
            
                    print('Epoch [%04d] : valid loss %e | accuracy %1.4e | time %1.2e sec.' % (val_loss, val_acc, t))
                else :
                    print('Epoch [%04d] : valid loss %e | accuracy None | time %1.2e sec.' % (epoch, val_loss, t))

        if plot is True :

            import matplotlib.pyplot as plt
            
            if len(_valid_loss) > 1:
                plt.plot(range(1,nepoch+1), _valid_loss, c='red')

            if len(_train_loss) > 1:
                plt.plot(range(1,nepoch+1), _train_loss, c='blue')
                plt.savefig('loss_epoch.png') 

            if len(_valid_acc) > 1:
                plt.plot(range(1,nepoch+1), _valid_acc, c='red')

            if len(_train_acc) > 1:
                plt.plot(range(1,nepoch+1), _train_acc, c='blue')
                plt.savefig('acc_epoch.png') 



    def Accuracy(self, prediction, target, reduce=True):
        '''
        Computes the accuracy for classification tasks
        
        prediction : tensor of torch.Size([batch_size, number of classes])
        The prediction tensor contains softmax activation function output
        i.e. probabilities of all classes
        Ex : tensor([[8.5442e-30, 1.0000e+00, 4.4564e-27, 1.4013e-45, 0.0000e+00],
            [8.7185e-10, 3.6507e-08, 1.0244e-05, 5.2405e-10, 9.9999e-01],
            [4.7920e-29, 1.0000e+00, 2.4772e-27, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])
        
        target : tensor of torch.Size([batch_size])
        Ex : tensor([1, 4, 0, 1])
        
        prediction.argmax(dim=1) returns the indices of the maximum values along the dim 1
        e.i. the class with the highest probability
        Ex : tensor([1, 4, 1, 1])
        
        (prediction.argmax(dim=1)==target) 
        compares the content of the two tensors and returns a True/False tensor 
        Ex : tensor([True, True, False, True])

        overlap = (prediction.argmax(dim=1)==target).sum() 
        counts the number of True booleans
        
        overlap/float(target.size()[-1])
        divides the number of True booleans by the batch size
        and thus returns an accuracy value
        Ex : tensor(0.7500)
        '''
        overlap = (prediction.argmax(dim=1)==target).sum()
        if reduce:
            return overlap/float(target.size()[-1])

        return overlap
   

    def format_output(self, out, acc, target):
        '''
        Format the network output depending on the task (classification/regression)
        '''
        if self.task == 'class':
            out = F.softmax(out, dim=1)
            target = torch.tensor([self.classes_idx[int(x)] for x in target])
            acc.append(self.Accuracy(out, target))

        else :
           out = out.reshape(-1)
           acc = None
        
        return out, acc, target

    
    def eval(self, loader):

        with self.model.eval():

            loss_func, loss_val = self.loss, 0
            out = []
            acc = []
            for data in loader:
                data = data.to(self.device)
                pred = self.model(data)
                pred, acc, data.y = self.format_output(pred, acc, data.y)

                loss_val += loss_func(pred, data.y)
                out += pred.reshape(-1).tolist()

        if self.task == 'class':
            return out, torch.mean(torch.stack(acc)), loss_val

        else :
            return out, acc, loss_val
            
            
    def _epoch(self, epoch):
        
        running_loss = 0
        acc = []
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            out, acc, data.y = self.format_output(out, acc, data.y)

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
