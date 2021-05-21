import numpy as np
from time import time
import h5py
import os

# torch import
import torch
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch_geometric.data import DataLoader

# deeprank_gnn import
from .DataSet import HDF5DataSet, DivideDataSet, PreCluster
from .Metrics import Metrics


class NeuralNet(object):

    def __init__(self, database, Net,
                 node_feature=['type', 'polarity', 'bsa'],
                 edge_feature=['dist'], target='irmsd', lr=0.01,
                 batch_size=32, percent=[1.0, 0.0],
                 database_eval=None, index=None, class_weights=None, task=None,
                 classes=[0, 1], threshold=4.0,
                 pretrained_model=None, shuffle=True, outdir='./', cluster_nodes='mcl'):
        """Class from which the network is trained, evaluated and tested

        Args:
            database (str, required): path(s) to hdf5 dataset(s). Unique hdf5 file or list of hdf5 files.
            Net (function, required): neural network function (ex. GINet, Foutnet etc.)
            node_feature (list, optional): type, charge, polarity, bsa (buried surface area), pssm,
                    cons (pssm conservation information), ic (pssm information content), depth,
                    hse (half sphere exposure).
                    Defaults to ['type', 'polarity', 'bsa'].
            edge_feature (list, optional): dist (distance). Defaults to ['dist'].
            target (str, optional): irmsd, lrmsd, fnat, capri_class, bin_class, dockQ.
                    Defaults to 'irmsd'.
            lr (float, optional): learning rate. Defaults to 0.01.
            batch_size (int, optional): defaults to 32.
            percent (list, optional): divides the input dataset into a training and an evaluation set.
                    Defaults to [1.0, 0.0].
            database_eval ([type], optional): independent evaluation set. Defaults to None.
            index ([type], optional): index of the molecules to consider. Defaults to None.
            class_weights ([list or bool], optional): weights provided to the cross entropy loss function.
                    The user can either input a list of weights or let DeepRanl-GNN (True) define weights
                    based on the dataset content. Defaults to None.
            task (str, optional): 'reg' for regression or 'class' for classification . Defaults to None.
            classes (list, optional): define the dataset target classes in classification mode. Defaults to [0, 1].
            threshold (int, optional): threshold to compute binary classification metrics. Defaults to 4.0.
            pretrained_model (str, optional): path to pre-trained model. Defaults to None.
            shuffle (bool, optional): shuffle the training set. Defaults to True.
            outdir (str, optional): output directory. Defaults to ./
            cluster_nodes (bool, optional): perform node clustering ('mcl' or 'louvain' algorithm). Default to 'mcl'.
        """
        # load the input data or a pretrained model
        # each named arguments is stored in a member vairable
        # i.e. self.node_feature = node_feature
        if pretrained_model is None:
            for k, v in dict(locals()).items():
                if k not in ['self', 'database', 'Net', 'database_eval']:
                    self.__setattr__(k, v)
            self.load_model(database, Net, database_eval)
            
            if self.task == None: 
                if self.target in ['irmsd', 'lrmsd', 'fnat', 'dockQ']:
                    self.task = 'reg'
                if self.target in ['bin_class', 'capri_classes']:
                    self.task = 'class'
                else: 
                    raise ValueError(
                        f"User target detected -> The task argument is required ('class' or 'reg'). \n\t"
                        f"Example: \n\t"
                        f""
                        f"model = NeuralNet(database, GINet,"
                        f"                  target='physiological_assembly',"
                        f"                  task='class',"
                        f"                  shuffle=True,"
                        f"                  percent=[0.8, 0.2])")
            
        else:
            self.load_params(pretrained_model)
            self.outdir = outdir
            self.target = target
            self.load_pretrained_model(database, Net)

    def load_pretrained_model(self, database, Net):
        """
        Loads pretrained model

        Args:
            database (str): path to hdf5 file(s)
            Net (function): neural network
        """
        # Load the test set
        test_dataset = HDF5DataSet(root='./', database=database,
                                   node_feature=self.node_feature, edge_feature=self.edge_feature,
                                   target=self.target)
        self.test_loader = DataLoader(
            test_dataset)
        PreCluster(test_dataset, method=self.cluster_nodes)

        print('Test set loaded')
        self.put_model_to_device(test_dataset, Net)

        self.set_loss()

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)

        # load the model and the optimizer state if we have one
        self.optimizer.load_state_dict(self.opt_loaded_state_dict)
        self.model.load_state_dict(self.model_load_state_dict)

    def load_model(self, database, Net, database_eval):
        """
        Loads model

        Args:
            database (str): path to the hdf5 file(s) of the training set
            Net (function): neural network
            database_eval (str): path to the hdf5 file(s) of the evaluation set

        Raises:
            ValueError: Invalid node clustering method.
        """
        # dataset
        dataset = HDF5DataSet(root='./', database=database, index=self.index,
                              node_feature=self.node_feature, edge_feature=self.edge_feature,
                              target=self.target)
        if self.cluster_nodes != None:
            if self.cluster_nodes == 'mcl' or self.cluster_nodes == 'louvain':
                PreCluster(dataset, method=self.cluster_nodes)
            else:
                raise ValueError(
                    f"Invalid node clustering method. \n\t"
                    f"Please set cluster_nodes to 'mcl', 'louvain' or None. Default to 'mcl' \n\t")

        # divide the dataset
        train_dataset, valid_dataset = DivideDataSet(
            dataset, percent=self.percent)

        # dataloader
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        if self.percent[1] > 0.0:
            self.valid_loader = DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        print('Training validation set loaded')

        # independent validation dataset
        if database_eval is not None:
            valid_dataset = HDF5DataSet(root='./', database=database_eval, index=self.index,
                                        node_feature=self.node_feature, edge_feature=self.edge_feature,
                                        target=self.target)
            self.valid_loader = DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            print('Independent validation set loaded')
            if self.cluster_nodes == 'mcl' or self.cluster_nodes == 'louvain':
                PreCluster(valid_dataset, method=self.cluster_nodes)

        else:
            print('No independent validation set loaded')

        self.put_model_to_device(dataset, Net)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)

        self.set_loss()

        # init lists
        self.train_acc = []
        self.train_loss = []

        self.valid_acc = []
        self.valid_loss = []

    def put_model_to_device(self, dataset, Net):
        """
        Puts the model on the available device

        Args:
            dataset (str): path to the hdf5 file(s)
            Net (function): neural network

        Raises:
            ValueError: Incorrect output shape
        """
        # get the device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # regression mode
        if self.task == 'reg':
            self.model = Net(dataset.get(
                0).num_features).to(self.device)
        # classification mode
        elif self.task == 'class':
            self.classes_to_idx = {i: idx for idx,
                                   i in enumerate(self.classes)}
            self.idx_to_classes = {idx: i for idx,
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

    def set_loss(self):
        """Sets the loss function (MSE loss for regression/ CrossEntropy loss for classification)."""
        if self.task == 'reg':
            self.loss = MSELoss()

        elif self.task == 'class':

            # assign weights to each class in case of unbalanced dataset
            self.weights = None
            if self.class_weights == True:
                targets_all = []
                for batch in self.train_loader:
                    targets_all.append(batch.y)

                targets_all = torch.cat(
                    targets_all).squeeze().tolist()
                self.weights = torch.tensor(
                    [targets_all.count(i) for i in self.classes], dtype=torch.float32)
                print('class occurences: {}'.format(self.weights))
                self.weights = 1.0 / self.weights
                self.weights = self.weights / self.weights.sum()
                print('class weights: {}'.format(self.weights))

            self.loss = nn.CrossEntropyLoss(
                weight=self.weights, reduction='mean')

    def train(self, nepoch=1, validate=False, save_model='last', hdf5='train_data.hdf5', save_epoch='intermediate', save_every=5):
        """
        Trains the model

        Args:
            nepoch (int, optional): number of epochs. Defaults to 1.
            validate (bool, optional): perform validation. Defaults to False.
            save_model (last, best, optional): save the model. Defaults to 'last'
            hdf5 (str, optional): hdf5 output file
            save_epoch (all, intermediate, optional)
            save_every (int, optional): save data every n epoch if save_epoch == 'intermediate'. Defaults to 5
        """
        # Output file name
        fname = self.update_name(hdf5, self.outdir)

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
            _acc = self.get_metrics('train', self.threshold).accuracy
            self.train_acc.append(_acc)

            # Print the loss and accuracy (training set)
            self.print_epoch_data(
                'train', epoch, _loss, _acc, t)

            # Validate the model
            if validate is True:

                t0 = time()
                _out, _y, _val_loss, self.data['eval'] = self.eval(
                    self.valid_loader)
                t = time() - t0

                self.valid_loss.append(_val_loss)
                self.valid_out = _out
                self.valid_y = _y
                _val_acc = self.get_metrics(
                    'eval', self.threshold).accuracy
                self.valid_acc.append(_val_acc)

                # Print loss and accuracy (validation set)
                self.print_epoch_data(
                    'valid', epoch, _val_loss, _val_acc, t)

                # save the best model (i.e. lowest loss value on validation data)
                if save_model == 'best':

                    if min(self.valid_loss) == _val_loss:
                        self.save_model(filename='t{}_y{}_b{}_e{}_lr{}_{}.pth.tar'.format(
                            self.task, self.target, str(self.batch_size), str(nepoch), str(self.lr), str(epoch)))

            else:
                # if no validation set, saves the best performing model on the traing set
                if save_model == 'best':
                    if min(self.train_loss) == _train_loss:
                        print(
                            'WARNING: The training set is used both for learning and model selection.')
                        print(
                            'this may lead to training set data overfitting.')
                        print(
                            'We advice you to use an external validation set.')
                        self.save_model(filename='t{}_y{}_b{}_e{}_lr{}_{}.pth.tar'.format(
                            self.task, self.target, str(self.batch_size), str(nepoch), str(self.lr), str(epoch)))

            # Save epoch data
            if (save_epoch == 'all') or (epoch == nepoch):
                self._export_epoch_hdf5(epoch, self.data)

            elif (save_epoch == 'intermediate') and (epoch % save_every == 0):
                self._export_epoch_hdf5(epoch, self.data)

        # Save the last model
        if save_model == 'last':
            self.save_model(filename='t{}_y{}_b{}_e{}_lr{}.pth.tar'.format(
                self.task, self.target, str(self.batch_size), str(nepoch), str(self.lr)))

        # Close output file
        self.f5.close()

    def test(self, database_test=None, threshold=4, hdf5='test_data.hdf5'):
        """
        Tests the model

        Args:
            database_test ([type], optional): test database
            threshold (int, optional): threshold use to tranform data into binary values. Defaults to 4.
            hdf5 (str, optional): output hdf5 file. Defaults to 'test_data.hdf5'.
        """
        # Output file name
        fname = self.update_name(hdf5, self.outdir)

        # Open output file for writting
        self.f5 = h5py.File(fname, 'w')

        # Loads the test dataset if provided
        if database_test is not None:
            # Load the test set
            test_dataset = HDF5DataSet(root='./', database=database_test,
                                            node_feature=self.node_feature, edge_feature=self.edge_feature,
                                            target=self.target)
            print('Test set loaded')
            PreCluster(test_dataset, method='mcl')

            self.test_loader = DataLoader(
                test_dataset)

        else:
            if self.load_pretrained_model == None:
                raise ValueError(
                    "You need to upload a test dataset \n\t"
                    "\n\t"
                    ">> model.test(test_dataset)\n\t"
                    "if a pretrained network is loaded, you can directly test the model on the loaded dataset :\n\t"
                    ">> model = NeuralNet(database_test, gnn, pretrained_model = model_saved, target=None)\n\t"
                    ">> model.test()\n\t")
        self.data = {}

        # Run test
        _out, _y, _test_loss, self.data['test'] = self.eval(
            self.test_loader)

        self.test_out = _out

        if len(_y) == 0:
            self.test_y = None
            self.test_acc = None
        else:
            self.test_y = _y
            _test_acc = self.get_metrics('test', threshold).accuracy
            self.test_acc = _test_acc

        self.test_loss = _test_loss
        self._export_epoch_hdf5(0, self.data)

        self.f5.close()

    def eval(self, loader):
        """
        Evaluates the model

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

        for data_batch in loader:
            data_batch = data_batch.to(self.device)
            pred = self.model(data_batch)
            pred, data_batch.y = self.format_output(
                pred, data_batch.y)

            # Check if a target value was provided (i.e. benchmarck scenario)
            if data_batch.y is not None:
                y += data_batch.y
                loss_val += loss_func(pred,
                                      data_batch.y).detach().item()
                # Save targets
                if self.task == 'class':
                    data['targets'] += [self.idx_to_classes(
                        x) for x in data_batch.y.numpy().tolist()]
                else:
                    data['targets'] += data_batch.y.numpy().tolist()

            # Get the outputs for export
            if self.task == 'class':
                pred = np.argmax(pred.detach(), axis=1)
            else:
                pred = pred.detach().reshape(-1)

            out += pred

            # Save predictions
            if self.task == 'class':
                data['outputs'] += [self.idx_to_classes(x)
                                    for x in pred.tolist()]
            else:
                data['outputs'] += pred.tolist()

            data['outputs'] += pred.tolist()

            # get the data
            data['mol'] += data_batch['mol']

        return out, y, loss_val, data

    def _epoch(self, epoch):
        """
        Runs a single epoch

        Returns:
            tuple: prediction, ground truth, running loss
        """
        running_loss = 0
        out = []
        y = []
        data = {'outputs': [], 'targets': [], 'mol': []}

        for data_batch in self.train_loader:

            data_batch = data_batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(data_batch)
            pred, data_batch.y = self.format_output(
                pred, data_batch.y)

            try:
                y += data_batch.y
            except ValueError:
                print(
                    "You must provide target values (y) for the training set")

            loss = self.loss(pred, data_batch.y)
            running_loss += loss.detach().item()
            loss.backward()
            self.optimizer.step()

            # get the outputs for export
            if self.task == 'class':
                pred = np.argmax(pred.detach(), axis=1)
            else:
                pred = pred.detach().reshape(-1)

            out += pred

            # save targets and predictions
            if self.task == 'class':
                data['targets'] += [self.idx_to_classes(x)
                                    for x in data_batch.y.numpy().tolist()]
                data['outputs'] += [self.idx_to_classes(x)
                                    for x in pred.tolist()]
            else:
                data['targets'] += data_batch.y.numpy().tolist()
                data['outputs'] += pred.tolist()

            # get the data
            data['mol'] += data_batch['mol']

        return out, y, running_loss, data

    def get_metrics(self, data='eval', threshold=4.0, binary=True):
        """
        Computes the metrics needed

        Args:
            data (str, optional): 'eval', 'train' or 'test'. Defaults to 'eval'.
            threshold (float, optional): threshold use to tranform data into binary values. Defaults to 4.0.
            binary (bool, optional): Transform data into binary data. Defaults to True.
        """
        if self.task == 'class':
            threshold = self.classes_to_idx[threshold]

        if data == 'eval':
            if len(self.valid_out) == 0:
                print('No evaluation set has been provided')

            else:
                pred = self.valid_out
                y = [x.item() for x in self.valid_y]

        elif data == 'train':
            if len(self.train_out) == 0:
                print('No training set has been provided')

            else:
                pred = self.train_out
                y = [x.item() for x in self.train_y]

        elif data == 'test':
            if len(self.test_out) == 0:
                print('No test set has been provided')

            if self.test_y == None:
                print(
                    'You must provide ground truth target values to compute the metrics')

            else:
                pred = self.test_out
                y = [x.item() for x in self.test_y]

        return Metrics(pred, y, self.target, threshold, binary)

    def compute_class_weights(self):

        targets_all = []
        for batch in self.train_loader:
            targets_all.append(batch.y)

        targets_all = torch.cat(targets_all).squeeze().tolist()
        weights = torch.tensor([targets_all.count(i)
                                for i in self.classes], dtype=torch.float32)
        print('class occurences: {}'.format(weights))
        weights = 1.0 / weights
        weights = weights / weights.sum()
        print('class weights: {}'.format(weights))
        return weights

    @staticmethod
    def print_epoch_data(stage, epoch, loss, acc, time):
        """
        Prints the data of each epoch

        Args:
            stage (str): tain or valid
            epoch (int): epoch number
            loss (float): loss during that epoch
            acc (float or None): accuracy
            time (float): timing of the epoch
        """
        if acc is None:
            acc_str = 'None'
        else:
            acc_str = '%1.4e' % acc

        print('Epoch [%04d] : %s loss %e | accuracy %s | time %1.2e sec.' % (epoch,
                                                                             stage, loss, acc_str, time))

    def format_output(self, pred, target=None):
        """Format the network output depending on the task (classification/regression)."""
        if self.task == 'class':
            pred = F.softmax(pred, dim=1)
            if target is not None:
                target = torch.tensor(
                    [self.classes_to_idx[int(x)] for x in target])

        else:
            pred = pred.reshape(-1)

        return pred, target

    @staticmethod
    def update_name(hdf5, outdir):
        """
        Checks if the file already exists, if so, update the name

        Args:
            hdf5 (str): hdf5 file
            outdir (str): output directory

        Returns:
            str: update hdf5 name
        """
        fname = os.path.join(outdir, hdf5)

        count = 0
        hdf5_name = hdf5.split('.')[0]

        # If file exists, change its name with a number
        while os.path.exists(fname):
            count += 1
            hdf5 = '{}_{:03d}.hdf5'.format(hdf5_name, count)
            fname = os.path.join(outdir, hdf5)

        return fname

    def plot_loss(self, name=''):
        """
        Plots the loss of the model as a function of the epoch

        Args:
            name (str, optional): name of the output file. Defaults to ''.
        """
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
        """
        Plots the accuracy of the model as a function of the epoch

        Args:
            name (str, optional): name of the output file. Defaults to ''.
        """
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
        """
        Plots the hitrate as a function of the models' rank

        Args:
            data (str, optional): which stage to consider train/eval/test. Defaults to 'eval'.
            threshold (int, optional): defines the value to split into a hit (1) or a non-hit (0). Defaults to 4.
            mode (str, optional): displays the hitrate as a number of hits ('count') or as a percentage ('percantage') . Defaults to 'percentage'.
        """
        import matplotlib.pyplot as plt

        try:

            hitrate = self.get_metrics(data, threshold).hitrate()

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

    def plot_scatter(self):
        """Scatters plot of the results."""
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
        """
        Saves the model to a file

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
                 'threshold': self.threshold,
                 'cluster_nodes': self.cluster_nodes}

        torch.save(state, filename)

    def load_params(self, filename):
        """
        Loads the parameters of a pretrained model

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
        self.cluster_nodes = state['cluster_nodes']

        self.opt_loaded_state_dict = state['optimizer']
        self.model_load_state_dict = state['model']

    def _export_epoch_hdf5(self, epoch, data):
        """
        Exports the epoch data to the hdf5 file.

        Exports the data of a given epoch in train/valid/test group.
        In each group are stored the predicted values (outputs),
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
