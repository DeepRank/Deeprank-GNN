import unittest
from graphprot.GraphGenMP import GraphHDF5

from .utils import PATH_TEST


def get_path(str):
    return (PATH_TEST / str).absolute().as_posix()


class TestCreateGraph(unittest.TestCase):

    def setUp(self):

        self.pdb_path = get_path('./data/pdb/1ATN/')
        self.pssm_path = get_path('./data/pssm/1ATN/')
        self.ref = get_path('./data/ref/1ATN/')

    def test_create(self):
        GraphHDF5(pdb_path=self.pdb_path, ref_path=self.ref, pssm_path=self.pssm_path,
                  graph_type='residue', outfile='1ATN_residue.hdf5',
                  nproc=1, tmpdir='./tmpdir')
