import glob 
import sys 
import time
import datetime 
import numpy as np

from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import GINet

pretrained_model = 'fold6_treg_yfnat_b128_e20_lr0.001_4.pt'
name = model_saved.split('/')[-1].split('.')[0]
gnn = GINet

### Change the path to your graphs
database_test = glob.glob('./test/*.hdf5')

### Make the prediction
start_time = time.time()
model = NeuralNet(database_test, gnn, pretrained_model = pretrained_model)    
model.test(threshold=None)
end_time = time.time()

print ('Elapsed time: {end_time-start_time}')

### If running DeepRank-GNN in a benchmarking scenario, uncomment those lines
#for threshold in np.arange(0.0, 1.0, 0.1):
#        test_metrics = model.get_metrics('test', threshold = threshold)
#        print('threshold', threshold, 'test accuracy:',
#              test_metrics.accuracy, 'TPR', test_metrics.sensitivity, 'TNR',
#              test_metrics.specificity, 'R2', test_metrics.r2_score)

