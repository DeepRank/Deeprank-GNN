import sys
import os
import torch
import numpy as np
from torch_geometric.data.dataset import Dataset
from torch_geometric.data.data import Data
from tqdm import tqdm
import h5py
import copy

from .community_pooling import community_detection, community_pooling


def DivideDataSet(dataset, percent=[0.8, 0.2], shuffle=True):
    """Divides the dataset into a training set and an evaluation set

    Args:
        dataset ([type])
        percent (list, optional): [description]. Defaults to [0.8, 0.2].
        shuffle (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    size = dataset.__len__()
    index = np.arange(size)

    if shuffle:
        np.random.shuffle(index)

    size1 = int(percent[0]*size)
    index1, index2 = index[:size1], index[size1:]

    dataset1 = copy.deepcopy(dataset)
    dataset1.index_complexes = [
        dataset.index_complexes[i] for i in index1]

    dataset2 = copy.deepcopy(dataset)
    dataset2.index_complexes = [
        dataset.index_complexes[i] for i in index2]

    return dataset1, dataset2


def PreCluster(dataset, method):
    """Pre-clusters nodes of the graphs

    Args:
        dataset (HDF5DataSet object)
        method (srt): 'mcl' (Markov Clustering) or 'louvain'
    """
    for fname, mol in tqdm(dataset.index_complexes):

        data = dataset.load_one_graph(fname, mol)

        if data is None:
            f5 = h5py.File(fname, 'a')
            try:
                print('deleting {}'.format(mol))
                del f5[mol]
            except:
                print('{} not found'.format(mol))
            f5.close()
            continue

        f5 = h5py.File(fname, 'a')
        grp = f5[mol]

        clust_grp = grp.require_group('clustering')

        if method.lower() in clust_grp:
            print('Deleting previous data for mol %s method %s' %
                  (mol, method))
            del clust_grp[method.lower()]

        method_grp = clust_grp.create_group(method.lower())

        cluster = community_detection(
            data.internal_edge_index, data.num_nodes, method=method)
        method_grp.create_dataset('depth_0', data=cluster)

        data = community_pooling(cluster, data)

        cluster = community_detection(
            data.internal_edge_index, data.num_nodes, method=method)
        method_grp.create_dataset('depth_1', data=cluster)

        f5.close()


class HDF5DataSet(Dataset):

    def __init__(self, root='./', database=None, transform=None, pre_transform=None,
                 dict_filter=None, target=None, tqdm=True, index=None,
                 node_feature='all', edge_feature=['dist'], clustering_method='mcl',
                 edge_feature_transform=lambda x: np.tanh(-x/2+2)+1):
        """Class from which the hdf5 datasets are loaded.

        Args:
            root (str, optional): [description]. Defaults to './'.
            database (str, optional): Path to hdf5 file(s). Defaults to None.
            transform (callable, optional): A function/transform that takes in an torch_geometric.data.Data object and returns a transformed version. The data object will be transformed before every access. Defaults to None.
            pre_transform (callable, optional):  A function/transform that takes in an torch_geometric.data.Data object and returns a transformed version. The data object will be transformed before being saved to disk.. Defaults to None.
            dict_filter dictionnary, optional): Dictionnary of type [name: cond] to filter the molecules. Defaults to None.
            target (str, optional): irmsd, lrmsd, fnat, bin, capri_class or DockQ. Defaults to None.
            tqdm (bool, optional): Show progress bar. Defaults to True.
            index (int, optional): index of a molecule. Defaults to None.
            node_feature (str or list, optional): consider all pre-computed node features ('all') or some defined node features (provide a list). Defaults to 'all'.
            edge_feature (list, optional): only distances are available in this version of DeepRank-GNN. Defaults to ['dist'].
            clustering_method (str, optional): 'mcl' (Markov Clustering) or 'louvain'. Defaults to 'mcl'.
            edge_feature_transform (function, optional): transformation applied to the edge features. Defaults to lambdax:np.tanh(-x/2+2)+1.
        """
        super().__init__(root, transform, pre_transform)

        # allow for multiple database
        self.database = database
        if not isinstance(database, list):
            self.database = [database]

        self.target = target
        self.dict_filter = dict_filter
        self.tqdm = tqdm
        self.index = index
        self.node_feature = node_feature
        self.edge_feature = edge_feature

        self.edge_feature_transform = edge_feature_transform

        self.clustering_method = clustering_method

        # check if the files are ok
        self.check_hdf5_files()

        # check the selection of features
        self.check_node_feature()
        self.check_edge_feature()

        # create the indexing system
        # alows to associate each mol to an index
        # and get fname and mol name from the index
        self.create_index_molecules()

    def len(self):
        """Gets the length of the dataset
        Returns:
            int: number of complexes in the dataset
        """
        return len(self.index_complexes)

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, index):
        """Gets one item from its unique index.

        Args:
            index (int): index of the complex
        Returns:
            dict: {'mol':[fname,mol],'feature':feature,'target':target}
        """

        fname, mol = self.index_complexes[index]
        data = self.load_one_graph(fname, mol)
        return data

    def check_hdf5_files(self):
        """Checks if the data contained in the hdf5 file is valid."""
        print("   Checking dataset Integrity")
        remove_file = []
        for fname in self.database:
            try:
                f = h5py.File(fname, 'r')
                mol_names = list(f.keys())
                if len(mol_names) == 0:
                    print('    -> %s is empty ' % fname)
                    remove_file.append(fname)
                f.close()
            except Exception as e:
                print(e)
                print('    -> %s is corrupted ' % fname)
                remove_file.append(fname)

        for name in remove_file:
            self.database.remove(name)

    def check_node_feature(self):
        """Checks if the required node features exist
        """
        f = h5py.File(self.database[0], 'r')
        mol_key = list(f.keys())[0]
        self.available_node_feature = list(
            f[mol_key+'/node_data/'].keys())
        f.close()

        if self.node_feature == 'all':
            self.node_feature = self.available_node_feature
        else:
            for feat in self.node_feature:
                if feat not in self.available_node_feature:
                    print(
                        feat, ' node feature not found in the file', self.database[0])
                    print('Possible node feature : ')
                    print('\n'.join(self.available_node_feature))
                    #raise ValueError('Feature Not found')
                    exit()

    def check_edge_feature(self):
        """Checks if the required edge features exist
        """
        f = h5py.File(self.database[0], 'r')
        mol_key = list(f.keys())[0]
        self.available_edge_feature = list(
            f[mol_key+'/edge_data/'].keys())
        f.close()

        if self.edge_feature == 'all':
            self.edge_feature = self.available_edge_feature
        elif self.edge_feature is not None:
            for feat in self.edge_feature:
                if feat not in self.available_edge_feature:
                    print(
                        feat, ' edge attribute not found in the file', self.database[0])
                    print('Possible edge attribute : ')
                    print('\n'.join(self.available_edge_feature))
                    #raise ValueError('Feature Not found')
                    exit()

    def load_one_graph(self, fname, mol):
        """Loads one graph

        Args:
            fname (str): hdf5 file name
            mol (str): name of the molecule

        Returns:
            Data object or None: torch_geometric Data object containing the node features, the internal and external edge features, the target and the xyz coordinates. Return None if features cannot be loaded.
        """
        f5 = h5py.File(fname, 'r')
        try:
            grp = f5[mol]
        except:
            f5.close()
            return None

        # nodes
        data = ()
        try:
            for feat in self.node_feature:
                vals = grp['node_data/'+feat][()]
                if vals.ndim == 1:
                    vals = vals.reshape(-1, 1)
                data += (vals,)
            x = torch.tensor(np.hstack(data), dtype=torch.float)

        except:
            print('node attributes not found in the file',
                  self.database[0])
            f5.close()
            return None

        try:
            # index ! we have to have all the edges i.e : (i,j) and (j,i)
            ind = grp['edge_index'][()]
            ind = np.vstack((ind, np.flip(ind, 1))).T
            edge_index = torch.tensor(
                ind, dtype=torch.long).contiguous()

            # edge feature (same issue than above)
            data = ()
            if self.edge_feature is not None:
                for feat in self.edge_feature:
                    vals = grp['edge_data/'+feat][()]
                    if vals.ndim == 1:
                        vals = vals.reshape(-1, 1)
                    data += (vals,)
                data = np.hstack(data)
                data = np.vstack((data, data))
                data = self.edge_feature_transform(data)
                edge_attr = torch.tensor(
                    data, dtype=torch.float).contiguous()

            else:
                edge_attr = None

            # internal edges
            ind = grp['internal_edge_index'][()]
            ind = np.vstack((ind, np.flip(ind, 1))).T
            internal_edge_index = torch.tensor(
                ind, dtype=torch.long).contiguous()

            # internal edge feature
            data = ()
            if self.edge_feature is not None:
                for feat in self.edge_feature:
                    vals = grp['internal_edge_data/'+feat][()]
                    if vals.ndim == 1:
                        vals = vals.reshape(-1, 1)
                    data += (vals,)
                data = np.hstack(data)
                data = np.vstack((data, data))
                data = self.edge_feature_transform(data)
                internal_edge_attr = torch.tensor(
                    data, dtype=torch.float).contiguous()

            else:
                internal_edge_attr = None

        except:
            print('edge features not found in the file',
                  self.database[0])
            f5.close()
            return None

        # target
        if self.target is None:
            y = None

        else:
            if grp['score/'+self.target][()] is not None:
                y = torch.tensor(
                    [grp['score/'+self.target][()]], dtype=torch.float).contiguous()
            else:
                y = None

        # pos
        pos = torch.tensor(grp['node_data/pos/']
                           [()], dtype=torch.float).contiguous()

        # load
        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    pos=pos)

        data.internal_edge_index = internal_edge_index
        data.internal_edge_attr = internal_edge_attr

        # mol name
        data.mol = mol

        # cluster
        if 'clustering' in grp.keys():
            if self.clustering_method in grp['clustering'].keys():
                if ('depth_0' in grp['clustering/{}'.format(self.clustering_method)].keys() and
                        'depth_1' in grp['clustering/{}'.format(
                            self.clustering_method)].keys()
                        ):
                    data.cluster0 = torch.tensor(
                        grp['clustering/' + self.clustering_method + '/depth_0'][()], dtype=torch.long)
                    data.cluster1 = torch.tensor(
                        grp['clustering/' + self.clustering_method + '/depth_1'][()], dtype=torch.long)
                else:
                    print('WARNING: no cluster detected')
            else:
                print('WARNING: no cluster detected')
        else:
            print('WARNING: no cluster detected')

        f5.close()
        return data

    def create_index_molecules(self):
        '''Creates the indexing of each molecule in the dataset.

        Creates the indexing: [ ('1ak4.hdf5,1AK4_100w),...,('1fqj.hdf5,1FGJ_400w)]
        This allows to refer to one complex with its index in the list
        '''
        print("   Processing data set")

        self.index_complexes = []

        desc = '{:25s}'.format('   Train dataset')
        if self.tqdm:
            data_tqdm = tqdm(
                self.database, desc=desc, file=sys.stdout)
        else:
            print('   Train dataset')
            data_tqdm = self.database
        sys.stdout.flush()

        for fdata in data_tqdm:
            if self.tqdm:
                data_tqdm.set_postfix(mol=os.path.basename(fdata))
            try:
                fh5 = h5py.File(fdata, 'r')
                if self.index is None:
                    mol_names = list(fh5.keys())
                else:
                    mol_names = [list(fh5.keys())[i]
                                 for i in self.index]
                for k in mol_names:
                    if self.filter(fh5[k]):
                        self.index_complexes += [(fdata, k)]
                fh5.close()
            except Exception as inst:
                print('\t\t--> Ignore File : ' + fdata)
                print(inst)

        self.ntrain = len(self.index_complexes)
        self.index_train = list(range(self.ntrain))
        self.ntot = len(self.index_complexes)

    def filter(self, molgrp):
        '''Filters the molecule according to a dictionary.

        The filter is based on the attribute self.dict_filter
        that must be either of the form: { 'name' : cond } or None

        Args:
            molgrp (str): group name of the molecule in the hdf5 file
        Returns:
            bool: True if we keep the complex False otherwise
        Raises:
            ValueError: If an unsuported condition is provided
        '''
        if self.dict_filter is None:
            return True

        for cond_name, cond_vals in self.dict_filter.items():

            try:
                val = molgrp[cond_name][()]
            except KeyError:
                print('   :Filter %s not found for mol %s' %
                      (cond_name, mol))

            # if we have a string it's more complicated
            if isinstance(cond_vals, str):

                ops = ['>', '<', '==']
                new_cond_vals = cond_vals
                for o in ops:
                    new_cond_vals = new_cond_vals.replace(o, 'val'+o)
                if not eval(new_cond_vals):
                    return False
            else:
                raise ValueError(
                    'Conditions not supported', cond_vals)

        return True
