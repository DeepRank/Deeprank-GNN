import unittest

from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import GINet
from deeprank_gnn.foutnet import FoutNet


__PLOT__ = False


class TestNeuralNet(unittest.TestCase):

    def setUp(self):
        self.database = 'tests/hdf5/1ATN_residue.hdf5'

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

        NN.train(nepoch=5, validate=False)

        NN.save_model('test.pth.tar')

        NN_cpy = NeuralNet(self.database, GINet,
                           pretrained_model='test.pth.tar')

        if __PLOT__:
            NN.plot_scatter()


if __name__ == "__main__":
    unittest.main()
