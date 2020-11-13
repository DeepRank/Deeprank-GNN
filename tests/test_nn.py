import unittest

from graphprot.NeuralNet import NeuralNet
from graphprot.ginet import GINet
from graphprot.foutnet import FoutNet


__PLOT__ = False


class TestNeuralNet(unittest.TestCase):

    def setUp(self):
        self.database = 'tests//hdf5/1ATN_residue.hdf5'

    def test_neural_net(self):
        NN = NeuralNet(self.database, GINet,
                       node_feature=['type', 'polarity', 'bsa',
                                     'depth', 'hse', 'ic', 'pssm'],
                       edge_feature=['dist'],
                       target='irmsd',
                       index=None,
                       task='reg',
                       batch_size=64,
                       percent=[0.8, 0.2])

        NN.train(nepoch=25, validate=False)

        if __PLOT__:
            NN.plot_scatter()
