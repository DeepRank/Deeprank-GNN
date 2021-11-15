import unittest
import numpy as np
from deeprank_gnn.tools.pssm_3dcons_to_deeprank import pssm_3dcons_to_deeprank
from deeprank_gnn.tools.hdf5_to_csv import hdf5_to_csv
from deeprank_gnn.tools.CustomizeGraph import add_target
from deeprank_gnn.tools.embedding import manifold_embedding


class TestTools(unittest.TestCase):

    def setUp(self):

        self.pdb_path = './tests/data/pdb/1ATN/'
        self.pssm_path = './tests/data/pssm/1ATN/1ATN.A.pdb.pssm'
        self.ref = './tests/data/ref/1ATN/'
        self.h5_train_ref = 'tests/data/train_ref/train_data.hdf5'

        self.h5_graphs = 'tests/hdf5/1ATN_residue.hdf5'

    def test_pssm_convert(self):
        pssm_3dcons_to_deeprank(self.pssm_path)

    def test_h52csv(self):
        hdf5_to_csv(self.h5_train_ref)

    def test_add_target(self):

        target_list = ''
        for i in range(1, 11):
            target_list += '1ATN_%dw %d\n' % (i, i)

        with open('target.lst', 'w') as f:
            f.write(target_list)

        add_target(self.h5_graphs, 'test_target', 'target.lst')

    def test_embeding(self):
        pos = np.random.rand(110, 3)
        for method in ['tsne', 'spectral', 'mds']:
            _ = manifold_embedding(pos, method=method)


if __name__ == "__main__":
    unittest.main()
