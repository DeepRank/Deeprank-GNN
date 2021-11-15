import unittest
from deeprank_gnn.tools.pssm_3dcons_to_deeprank import pssm_3dcons_to_deeprank


class TestTools(unittest.TestCase):

    def setUp(self):

        self.pdb_path = './tests/data/pdb/1ATN/'
        self.pssm_path = './tests/data/pssm/1ATN/1ATN.A.pdb.pssm'
        self.ref = './tests/data/ref/1ATN/'

    def test_pssm_convert(self):
        pssm_3dcons_to_deeprank(self.pssm_path)


if __name__ == "__main__":
    unittest.main()
