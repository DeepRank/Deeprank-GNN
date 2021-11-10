import unittest
from deeprank_gnn.GraphGenMP import GraphHDF5


class TestCreateGraph(unittest.TestCase):

    def setUp(self):

        self.pdb_path = './tests/data/pdb/1ATN/'
        self.pssm_path = './tests/data/pssm/1ATN/'
        self.ref = './tests/data/ref/1ATN/'

    def test_create(self):
        GraphHDF5(pdb_path=self.pdb_path, ref_path=self.ref, pssm_path=self.pssm_path,
                  graph_type='residue', outfile='1ATN_residue.hdf5',
                  nproc=1, tmpdir='./tmpdir')
