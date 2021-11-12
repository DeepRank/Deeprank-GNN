import unittest

from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import GINet
from deeprank_gnn.foutnet import FoutNet
from deeprank_gnn.sGAT import sGAT


def _model_base_test(database, model, plot):

    NN = NeuralNet(database, model,
                   node_feature=['type', 'polarity', 'bsa',
                                 'depth', 'hse', 'ic', 'pssm'],
                   edge_feature=['dist'],
                   target='irmsd',
                   index=None,
                   task='reg',
                   batch_size=64,
                   percent=[0.8, 0.2])

    NN.train(nepoch=5, validate=True)

    NN.save_model('test.pth.tar')

    NN_cpy = NeuralNet(database, model,
                       pretrained_model='test.pth.tar')

    if plot:
        NN.plot_scatter()
        NN.plot_loss()
        NN.plot_acc()
        NN.plot_hit_rate()


class TestNeuralNet(unittest.TestCase):

    def setUp(self):
        self.database = 'tests/hdf5/1ATN_residue.hdf5'

    def test_ginet(self):
        _model_base_test(self.database, GINet, True)

    def test_fout(self):
        _model_base_test(self.database, FoutNet, False)

    def test_sgat(self):
        _model_base_test(self.database, sGAT, False)


if __name__ == "__main__":
    unittest.main()
