import unittest
from deeprank_gnn.tools.pssm_3dcons_to_deeprank import pssm_3dcons_to_deeprank
from deeprank_gnn.tools.hdf5_to_csv import hdf5_to_csv


class TestTools(unittest.TestCase):

    def setUp(self):

        self.pdb_path = './tests/data/pdb/1ATN/'
        self.pssm_path = './tests/data/pssm/1ATN/1ATN.A.pdb.pssm'
        self.ref = './tests/data/ref/1ATN/'
        self.h5_train_ref = 'tests/data/train_ref/train_data.hdf5'

    def test_pssm_convert(self):
        pssm_3dcons_to_deeprank(self.pssm_path)

    def test_h52csv(self):
        hdf5_to_csv(self.h5_train_ref)


if __name__ == "__main__":
    unittest.main()
