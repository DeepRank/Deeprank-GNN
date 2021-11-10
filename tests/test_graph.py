
import unittest
import h5py

from deeprank_gnn.Graph import Graph


class TestGraph(unittest.TestCase):

    def setUp(self):
        self.graph = Graph()
        self.graph.h52nx('tests/hdf5/1ATN_residue.hdf5', '1ATN_1w')

    def test_plot_2d(self):
        self.graph.plotly_2d('1ATN', disable_plot=True)

    def test_plot_3d(self):
        self.graph.plotly_3d('1ATN', disable_plot=True)


if __name__ == "__main__":
    unittest.main()
