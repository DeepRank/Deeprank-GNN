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
from .DataSet import HDF5DataSet, DivideDataSet, PreCluster
from .Metrics import Metrics


class NeuralNet(object):

    def __init__(self, database, Net,
                 node_feature=['type', 'polarity', 'bsa'],
                 edge_feature=['dist'], target='irmsd',
                 batch_size=32, percent=[0.8, 0.2], index=None, database_eval=None,
                 class_weights=None, task='class', classes=[0, 1], threshold=4):

        # dataset
        dataset = HDF5DataSet(root='./', database=database, index=index,
                              node_feature=node_feature, edge_feature=edge_feature,
                              target=target)
        #PreCluster(dataset, method='mcl')

        train_dataset, valid_dataset = DivideDataSet(
            dataset, percent=percent)

        # independent validation dataset
        if database_eval is not None:
            valid_dataset = HDF5DataSet(root='./', database=database_eval, index=index,
                                        node_feature=node_feature, edge_feature=edge_feature,
                                        target=target)
            print('Independent validation set loaded')
            #PreCluster(valid_dataset, method='mcl')

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
        self.threshold = threshold

        # put the model
        if self.task == 'reg':
            self.model = Net(dataset.get(
                0).num_features).to(self.device)

        elif self.task == 'class':
            self.classes = classes
            self.classes_idx = {i: idx for idx,
                                i in enumerate(self.classes)}
            self.output_shape = len(self.classes)
            try:
                self.model = Net(dataset.get(
                    0).num_features, self.output_shape).to(self.device)
            except:
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
            self.loss = nn.CrossEntropyLoss(
                weight=self.class_weights, reduction='mean')

        # init lists
        self.train_acc = []
        self.train_loss = []

        self.valid_acc = []
        self.valid_loss = []

    def plot_loss(self):

        nepoch = self.nepoch
        train_loss = self.train_loss
        valid_loss = self.valid_loss

        import matplotlib.pyplot as plt

        if len(valid_loss) > 1:
            plt.plot(range(1, nepoch+1), valid_loss,
                     c='red', label='valid')

        if len(train_loss) > 1:
            plt.plot(range(1, nepoch+1), train_loss,
                     c='blue', label='train')
            plt.title("Loss/ epoch")
            plt.xlabel("Number of epoch")
            plt.ylabel("Total loss")
            plt.legend()
            plt.savefig('loss_epoch.png')
            plt.close()

    def plot_acc(self):

        nepoch = self.nepoch
        train_acc = self.train_acc
        valid_acc = self.valid_acc

        import matplotlib.pyplot as plt

        if len(valid_acc) > 1:
            plt.plot(range(1, nepoch+1), valid_acc,
                     c='red', label='valid')

        if len(train_acc) > 1:
            plt.plot(range(1, nepoch+1), train_acc,
                     c='blue', label='train')
            plt.title("Accuracy/ epoch")
            plt.xlabel("Number of epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig('acc_epoch.png')
            plt.close()

    def plot_hit_rate(self, data='eval', threshold=4, mode='percentage'):
        '''
        Plots the hitrate as a function of the models' rank

        - threshold (default = 4): defines the value to split into a hit (1) or a non-hit (0)
        - mode : displays the hitrate as a number of hits ('count') or as a percentage ('percantage')
        '''

        import matplotlib.pyplot as plt

        try:

            hitrate = self.get_metrics(data, threshold).HitRate()

            nb_models = len(hitrate)
            X = range(1, nb_models + 1)

            if mode == 'percentage':
                hitrate = [x/nb_models for x in hitrate]

            plt.plot(X, hitrate, c='blue', label='train')
            plt.title("Hit rate")
            plt.xlabel("Number of models")
            plt.ylabel("Hit Rate")
            plt.legend()
            plt.savefig('hitrate.png')
            plt.close()

        except:
            print('No hit rate plot could be generated for you {} task'.format(
                self.task))

    def train(self, nepoch=1, validate=False, plot=False):

        self.nepoch = nepoch

        for epoch in range(1, nepoch+1):

            self.model.train()

            t0 = time()
            _out, _y, _loss = self._epoch(epoch)
            t = time() - t0
            self.train_loss.append(_loss)
            self.train_out = _out
            self.train_y = _y
            _acc = self.get_metrics('train', self.threshold).ACC
            self.train_acc.append(_acc)

            self.print_epoch_data('train', epoch, _loss, _acc, t)

            if validate is True:
                t0 = time()
                _out, _y, _val_loss = self.eval(self.valid_loader)
                t = time() - t0

                self.valid_loss.append(_val_loss)
                self.valid_out = _out
                self.valid_y = _y
                _val_acc = self.get_metrics(
                    'eval', self.threshold).ACC
                self.valid_acc.append(_val_acc)

                self.print_epoch_data(
                    'valid', epoch, _val_loss, _val_acc, t)

    @staticmethod
    def print_epoch_data(stage, epoch, loss, acc, time):
        """print the data of each epoch

        Args:
            stage (str): tain or valid
            epoch (int): epoch number
            loss (float): loss during that epoch
            acc (float or None): accuracy
            time (float): tiing of the epoch
        """

        if acc is None:
            acc_str = 'None'
        else:
            acc_str = '%1.4e' % acc

        print('Epoch [%04d] : %s loss %e | accuracy %s | time %1.2e sec.' % (epoch,
                                                                             stage, loss, acc_str, time))

    def format_output(self, out, target):
        '''
        Format the network output depending on the task (classification/regression)
        '''
        if self.task == 'class':
            out = F.softmax(out, dim=1)
            target = torch.tensor(
                [self.classes_idx[int(x)] for x in target])

        else:
            out = out.reshape(-1)

        return out, target

    def test(self, database_test, threshold):

        test_dataset = HDF5DataSet(root='./', database=database_test,
                                        node_feature=self.node_feature, edge_feature=self.edge_feature,
                                        target=self.target)
        print('Test set loaded')
        PreCluster(test_dataset, method='mcl')

        self.test_loader = DataLoader(
            test_dataset)

        _out, _y, _test_loss = self.eval(self.test_loader)

        self.test_out = _out
        self.test_y = _y
        _test_acc = self.get_metrics('test', threshold).ACC
        self.test_acc = _test_acc
        self.test_loss = _test_loss

    def eval(self, loader):

        self.model.eval()

        loss_func, loss_val = self.loss, 0
        out = []
        acc = []
        y = []
        for data in loader:
            data = data.to(self.device)
            pred = self.model(data)
            pred, data.y = self.format_output(pred, data.y)

            y += data.y
            loss_val += loss_func(pred, data.y)
            out += pred.reshape(-1).tolist()

        if self.task == 'class':
            return out, y, loss_val

        else:
            return out, y, loss_val

    def _epoch(self, epoch):

        running_loss = 0
        out = []
        y = []
        for data in self.train_loader:

            data = data.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(data)
            pred, data.y = self.format_output(pred, data.y)

            y += data.y
            loss = self.loss(pred, data.y)
            running_loss += loss.data.item()
            loss.backward()
            out += pred.reshape(-1).tolist()
            self.optimizer.step()

        if self.task == 'class':
            return out, y, running_loss

        else:
            return out, y, running_loss

    def get_metrics(self, data='eval', threshold=4, binary=True):

        if data == 'eval':
            if len(self.valid_out) == 0:
                print('No evaluation set has been provided')

            pred = self.valid_out
            y = [x.item() for x in self.valid_y]

        elif data == 'train':
            if len(self.train_out) == 0:
                print('No training set has been provided')

            pred = self.train_out
            y = [x.item() for x in self.train_y]

        elif data == 'test':
            if len(self.test_out) == 0:
                print('No test set has been provided')

            pred = self.test_out
            y = [x.item() for x in self.test_y]

        return Metrics(pred, y, self.target, threshold, binary)

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
        """Load model from a saved file

        Args:
            filename (str): name of the file to be loaded
        """
        print('Loading file : %s' % filename)
        state = torch.load(filename)

        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.node_feature = state['node']
        self.edge_feature = state['edge']
        self.target = state['target']
