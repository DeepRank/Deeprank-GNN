import numpy as np
from tqdm import tqdm
from time import time
import h5py
import os 

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
                 edge_feature=['dist'], target='irmsd', lr=0.01,
                 batch_size=32, percent=[0.8, 0.2], index=None, database_eval=None,
                 class_weights=None, task='class', classes=[0, 1], threshold=4,
                 pretrained_model=None, shuffle=False, outdir='./'):

        # load the input data or a pretrained model
        # each named arguments is stored in a member vairable
        # i.e. self.node_feature = node_feature
        if pretrained_model is None:
            for k, v in dict(locals()).items():
                if k not in ['self', 'database', 'Net', 'database_eval']:
                    self.__setattr__(k, v)

        else:
            self.load_params(pretrained_model)
            self.outdir = outdir
        # dataset
        dataset = HDF5DataSet(root='./', database=database, index=self.index,
                              node_feature=self.node_feature, edge_feature=self.edge_feature,
                              target=self.target)
        # PreCluster(dataset, method='mcl')


        # divide the dataset
        train_dataset, valid_dataset = DivideDataSet(
            dataset, percent=self.percent)

        # dataloader
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        # independent validation dataset
        if database_eval is not None:
            valid_dataset = HDF5DataSet(root='./', database=database_eval, index=self.index,
                                        node_feature=self.node_feature, edge_feature=self.edge_feature,
                                        target=self.target)
            self.valid_loader = DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            print('Independent validation set loaded')
            # PreCluster(valid_dataset, method='mcl')
            
        else:
            print('No independent validation set loaded')

        # get the device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

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
            self.model.parameters(), lr=self.lr)

        # laod the optimizer state if we have one
        if pretrained_model is not None:
            self.optimizer.load_state_dict(self.opt_loaded_state_dict)

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

    def plot_loss(self, name=''):
        """Plot the loss of the model."""

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
            plt.savefig('loss_epoch{}.png'.format(name))
            plt.close()

    def plot_acc(self, name=''):
        """Plot the accuracy of the model."""

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
            plt.savefig('acc_epoch{}.png'.format(name))
            plt.close()

    def plot_hit_rate(self, data='eval', threshold=4, mode='percentage', name=''):
        """Plots the hitrate as a function of the models' rank

        Args:
            data (str, optional): which stage to consider train/eval/test. Defaults to 'eval'.
            threshold (int, optional): defines the value to split into a hit (1) or a non-hit (0). Defaults to 4.
            mode (str, optional): displays the hitrate as a number of hits ('count') or as a percentage ('percantage') . Defaults to 'percentage'.
        """

        import matplotlib.pyplot as plt

        try:

            hitrate = self.get_metrics(data, threshold).HitRate()

            nb_models = len(hitrate)
            X = range(1, nb_models + 1)

            if mode == 'percentage':
                hitrate /= hitrate.sum()

            plt.plot(X, hitrate, c='blue', label='train')
            plt.title("Hit rate")
            plt.xlabel("Number of models")
            plt.ylabel("Hit Rate")
            plt.legend()
            plt.savefig('hitrate{}.png'.format(name))
            plt.close()

        except:
            print('No hit rate plot could be generated for you {} task'.format(
                self.task))

    def update_name(hdf5):
        """Check if the file already exists
        if so, update the name
        ex. 1: train.hdf5 -> train_001.hdf5
        ex. 2: train_001.hdf5 -> train_002.hdf5 
        """

        fname = os.path.join(self.outdir, hdf5)

        count = 0
        hdf5_name = hdf5.split('.')[0]

        # If file exists, change its name with a number                                                                                               
        while os.path.exists(name) : 
            count += 1
            hdf5 = '{}_{:03d}.hdf5'.format(hdf5_name, count)
            fname = os.path.join(self.outdir, hdf5)

        return fname

    def train(self, nepoch=1, validate=False, plot=False, save_model='last', hdf5='train_data.hdf5', save_epoch='intermediate', save_every=5):
        """Train the model

        Args:
            nepoch (int, optional): number of epochs. Defaults to 1.
            validate (bool, optional): perform validation. Defaults to False.
            plot (bool, optional): plot the results. Defaults to False.
            save_model (last, best, optional): save the model. Defaults to 'last' 
            hdf5 (str, optional): hdf5 output file
            save_epoch (all, intermediate, optional)
            save_every (int, optional): save data every n epoch if save_epoch == 'intermediate'. Defaults to 5
        """

        # Output file name
        fname = update_name(hdf5)
        
        # Open output file for writting
        self.f5 = h5py.File(fname, 'w')

        # Number of epochs
        self.nepoch = nepoch

        # Loop over epochs
        self.data = {}
        for epoch in range(1, nepoch+1):

            # Train the model
            self.model.train()

            t0 = time()
            _out, _y, _loss, self.data['train'] = self._epoch(epoch)
            t = time() - t0
            self.train_loss.append(_loss)
            self.train_out = _out
            self.train_y = _y
            _acc = self.get_metrics('train', self.threshold).ACC
            self.train_acc.append(_acc)

            # Print the loss and accuracy (training set)
            self.print_epoch_data(
                'train', epoch, _loss, _acc, t)

            # Validate the model
            if validate is True:

                t0 = time()
                _out, _y, _val_loss, self.data['eval'] = self.eval(self.valid_loader)
                t = time() - t0

                self.valid_loss.append(_val_loss)
                self.valid_out = _out
                self.valid_y = _y
                _val_acc = self.get_metrics(
                    'eval', self.threshold).ACC
                self.valid_acc.append(_val_acc)

                # Print loss and accuracy (validation set)
                self.print_epoch_data(
                    'valid', epoch, _val_loss, _val_acc, t)

                # save the best model (i.e. lowest loss value on validation data)
                if save_model == 'best' :
                    
                    if min(self.valid_loss) == _val_loss :
                        self.save_model(filename='t{}_y{}_b{}_e{}_lr{}.pth.tar'.format(
                            self.task, self.target, str(self.batch_size), str(nepoch), str(self.lr)))

            else :
                # if no validation set, saves the best performing model on the traing set 
                if save_model == 'best' :
                    if min(self.train_loss) == _train_loss :
                        print ('WARNING: The training set is used both for learning and model selection.')
                        print('this may lead to training set data overfitting.')
                        print('We advice you to use an external validation set.')
                        self.save_model(filename='t{}_y{}_b{}_e{}_lr{}.pth.tar'.format(
                            self.task, self.target, str(self.batch_size), str(nepoch), str(self.lr)))

            # Save epoch data
            if (save_epoch == 'all') or (epoch == nepoch) :
                self._export_epoch_hdf5(epoch, self.data)
            
            elif (save_epoch == 'intermediate') and (epoch%save_every == 0) :
                self._export_epoch_hdf5(epoch, self.data)

        # Save the last model 
        if save_model == 'last' :
            self.save_model(filename='t{}_y{}_b{}_e{}_lr{}.pth.tar'.format(self.task, self.target, str(self.batch_size), str(nepoch), str(self.lr)))
        
        # Close output file
        self.f5.close()


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
        """Format the network output depending on the task (classification/regression)."""

        if self.task == 'class':
            out = F.softmax(out, dim=1)
            target = torch.tensor(
                [self.classes_idx[int(x)] for x in target])

        else:
            out = out.reshape(-1)

        return out, target

    def test(self, database_test, threshold=4, hdf5='test_data.hdf5'):
        """Test the model

        Args:
            database_test ([type]): [description]
            threshold ([type]): [description]
        """

        # Output file 
        # Output file name
        fname = update_name(hdf5)
        
        # Open output file for writting
        self.f5 = h5py.File(fname, 'w')

        # Load the test set
        test_dataset = HDF5DataSet(root='./', database=database_test,
                                        node_feature=self.node_feature, edge_feature=self.edge_feature,
                                        target=self.target)
        print('Test set loaded')
        PreCluster(test_dataset, method='mcl')

        self.test_loader = DataLoader(
            test_dataset)

        self.data = {}

        # Run test
        _out, _y, _test_loss, self.data['test'] = self.eval(self.test_loader)

        self.test_out = _out
        self.test_y = _y
        _test_acc = self.get_metrics('test', threshold).ACC
        self.test_acc = _test_acc
        self.test_loss = _test_loss

        self._export_epoch_hdf5(0, self.data)
            
        self.f5.close()

    def eval(self, loader):
        """Evaluate the model

        Args:
            loader (DataLoader): [description]

        Returns:
            (tuple):
        """

        self.model.eval()

        loss_func, loss_val = self.loss, 0
        out = []
        y = []
        data = {'outputs': [], 'targets': [], 'mol': []}

        for d in loader:
            d = d.to(self.device)
            pred = self.model(d)
            pred, d.y = self.format_output(pred, d.y)

            y += d.y
            loss_val += loss_func(pred, d.y).detach().item()
            out += pred.reshape(-1).tolist()

            # get the outputs for export
            data['outputs'] += pred.reshape(-1).tolist()
            data['targets'] += d.y.numpy().tolist()

            # get the data
            data['mol'] += d['mol']

        return out, y, loss_val, data

    def _epoch(self, epoch):
        """Run a single epoch

        Returns:
            tuple: prediction, ground truth, running loss
        """

        running_loss = 0
        out = []
        y = []
        data = {'outputs': [], 'targets': [], 'mol': []}        

        for d in self.train_loader:

            d = d.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(d)
            pred, d.y = self.format_output(pred, d.y)

            y += d.y
            loss = self.loss(pred, d.y)
            running_loss += loss.detach().item()
            loss.backward()
            out += pred.reshape(-1).tolist()
            self.optimizer.step()

            # get the outputs for export
            data['outputs'] += pred.reshape(-1).tolist()
            data['targets'] += d.y.numpy().tolist()

            # get the data
            data['mol'] += d['mol']

        return out, y, running_loss, data

    def get_metrics(self, data='eval', threshold=4, binary=True):
        """Compute the metrics needed

        Args:
            data (str, optional): [description]. Defaults to 'eval'.
            threshold (int, optional): [description]. Defaults to 4.
            binary (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """

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
        """Scatter plot of the results"""
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
        """Save the model to a file

        Args:
            filename (str, optional): name of the file. Defaults to 'model.pth.tar'.
        """

        state = {'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'node': self.node_feature,
                 'edge': self.edge_feature,
                 'target': self.target,
                 'task': self.task,
                 'classes': self.classes,
                 'class_weight': self.class_weights,
                 'batch_size': self.batch_size,
                 'percent': self.percent,
                 'lr': self.lr,
                 'index': self.index,
                 'shuffle': self.shuffle,
                 'threshold': self.threshold}

        torch.save(state, filename)

    def load_params(self, filename):
        """Load the parameters of a rpetrained model

        Args:
            filename ([type]): [description]

        Returns:
            [type]: [description]
        """

        state = torch.load(filename)

        self.node_feature = state['node']
        self.edge_feature = state['edge']
        self.target = state['target']
        self.batch_size = state['batch_size']
        self.percent = state['percent']
        self.lr = state['lr']
        self.index = state['index']
        self.class_weights = state['class_weight']
        self.task = state['task']
        self.classes = state['classes']
        self.threshold = state['threshold']
        self.shuffle = state['shuffle']

        self.opt_loaded_state_dict = state['optimizer']

    def _export_epoch_hdf5(self, epoch, data):
        """Export the epoch data to the hdf5 file.
        Export the data of a given epoch in train/valid/test group.
        In each group are stored the predcited values (outputs),
        ground truth (targets) and molecule name (mol).
        Args:
            epoch (int): index of the epoch
            data (dict): data of the epoch
        """

        # create a group
        grp_name = 'epoch_%04d' % epoch
        grp = self.f5.create_group(grp_name)

        grp.attrs['task'] = self.task
        grp.attrs['target'] = self.target
        grp.attrs['batch_size'] = self.batch_size

        # loop over the pass_type : train/valid/test
        for pass_type, pass_data in data.items():

            # we don't want to breack the process in case of issue
            try:

                # create subgroup for the pass
                sg = grp.create_group(pass_type)

                # loop over the data : target/output/molname
                for data_name, data_value in pass_data.items():

                    # mol name is a bit different
                    # since there are strings
                    if data_name == 'mol':
                        data_value = np.string_(data_value)
                        string_dt = h5py.special_dtype(vlen=str)
                        sg.create_dataset(
                            data_name, data=data_value, dtype=string_dt)

                    # output/target values
                    else:
                        sg.create_dataset(data_name, data=data_value)

            except TypeError:
                raise ValueError("Error in export epoch to hdf5")
