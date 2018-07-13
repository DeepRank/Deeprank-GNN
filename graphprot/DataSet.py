import sys, os
import torch
import numpy as np
from torch_geometric.data.dataset import Dataset
from torch_geometric.data.data import Data
from tqdm import tqdm
import h5py

class HDF5DataSet(Dataset):

    def __init__(self,root,database=None,transform=None,pre_transform=None,
                dict_filter = None, target='dockQ',tqdm = True):
        super().__init__(root,transform,pre_transform)

        # allow for multiple database
        self.database = database
        if not isinstance(database,list):
            self.database = [database]

        self.target = target
        self.dict_filter = dict_filter
        self.tqdm = tqdm

        # check if the files are ok
        self.check_hdf5_files()

        # create the indexing system
        # alows to associate each mol to an index
        # and get fname and mol name from the index
        self.create_index_molecules()


    def __len__(self):
        """Get the length of the dataset
        Returns:
            int: number of complexes in the dataset
        """
        return len(self.index_complexes)


    def _download(self):
        pass

    def _process(self):
        pass

    def get(self,index):
        """Get one item from its unique index.
        Args:
            index (int): index of the complex
        Returns:
            dict: {'mol':[fname,mol],'feature':feature,'target':target}
        """

        fname,mol = self.index_complexes[index]
        data = self.load_one_graph(fname,mol)
        return data


    def check_hdf5_files(self):
        """Check if the data contained in the hdf5 file is ok."""

        print("   Checking dataset Integrity")
        remove_file = []
        for fname in self.database:
            try:
                f = h5py.File(fname,'r')
                mol_names = list(f.keys())
                if len(mol_names) == 0:
                    print('    -> %s is empty ' %fname)
                    remove_file.append(fname)
                f.close()
            except Exception as e:
                print(e)
                print('    -> %s is corrputed ' %fname)
                remove_file.append(fname)

        for name in remove_file:
            self.database.remove(name)


    def load_one_graph(self,fname,mol):

        f5 = h5py.File(fname,'r')
        grp = f5[mol]

        #nodes
        x = torch.tensor(grp['node'].value,dtype=torch.float)

        # index ! we have to have all the edges i.e : (i,j) and (j,i)
        ind = grp['edge_index'].value
        ind = np.vstack((ind,np.flip(ind,1))).T
        edge_index = torch.tensor(ind,dtype=torch.long)

        # edge attr (same issue than above)
        attr = grp['edge_attr'].value
        attr = np.vstack((attr,attr))
        edge_attr = torch.tensor(attr,dtype=torch.float)

        #target
        y = torch.tensor([grp[self.target].value],dtype=torch.float)

        #pos
        pos = torch.tensor(grp['pos'].value,dtype=torch.float)

        # load
        data = Data(x = x,
                    edge_index = edge_index,
                    edge_attr = edge_attr,
                    y = y,
                    pos=pos)

        f5.close()
        return data



    def create_index_molecules(self):
        '''Create the indexing of each molecule in the dataset.
        Create the indexing: [ ('1ak4.hdf5,1AK4_100w),...,('1fqj.hdf5,1FGJ_400w)]
        This allows to refer to one complex with its index in the list
        '''
        print("   Processing data set")

        self.index_complexes = []

        desc = '{:25s}'.format('   Train dataset')
        if self.tqdm:
            data_tqdm = tqdm(self.database,desc=desc,file=sys.stdout)
        else:
            print('   Train dataset')
            data_tqdm = self.database
        sys.stdout.flush()

        for fdata in data_tqdm:
            if self.tqdm:
                data_tqdm.set_postfix(mol=os.path.basename(fdata))
            try:
                fh5 = h5py.File(fdata,'r')
                mol_names = list(fh5.keys())
                for k in mol_names:
                    if self.filter(fh5[k]):
                        self.index_complexes += [(fdata,k)]
                fh5.close()
            except Exception as inst:
                print('\t\t-->Ignore File : ' + fdata)
                print(inst)

        self.ntrain = len(self.index_complexes)
        self.index_train = list(range(self.ntrain))
        self.ntot = len(self.index_complexes)

    def filter(self,molgrp):
        '''Filter the molecule according to a dictionary.
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

        for cond_name,cond_vals in self.dict_filter.items():

            try:
                val = molgrp[cond_name].value
            except KeyError:
                print('   :Filter %s not found for mol %s' %(cond_name,mol))

            # if we have a string it's more complicated
            if isinstance(cond_vals,str):

                ops = ['>','<','==']
                new_cond_vals = cond_vals
                for o in ops:
                    new_cond_vals = new_cond_vals.replace(o,'val'+o)
                if not eval(new_cond_vals):
                    return False
            else:
                raise ValueError('Conditions not supported', cond_vals)

        return True


if __name__ == '__main__':

    dataset = HDF5DataSet(root='./',database='graph_residue.hdf5')
    #dataset = HDF5DataSet(root='./',database='graph_residue.hdf5')
    dataset.get(1)