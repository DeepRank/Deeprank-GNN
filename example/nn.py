from graphprot.NeuralNet import NeuralNet
from graphprot.ginet import GINet
from graphprot.foutnet import FoutNet

database = './hdf5/1ACB_residue.hdf5'

NN = NeuralNet(database, GINet,
               node_feature=['type', 'polarity', 'bsa',
                             'depth', 'hse', 'ic', 'pssm'],
               edge_feature=['dist'],
               target='irmsd',
               index=range(400),
               batch_size=64,
               percent=[0.8, 0.2])

NN.train(nepoch=250, validate=False)
NN.plot_scatter()
