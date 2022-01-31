from deeprank_gnn.GraphGenMP import GraphHDF5
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import GINet
from deeprank_gnn.tools.CustomizeGraph import add_target
import glob

#path to the docking models in pdb format
pdb_path = '../DC/pdb/' 
#path to the pssm files
pssm_path = '../DC/pssm/'

#path to the graphs (hdf5) file 
database_test = 'biological_vs_crystal.hdf5'

# Generation of the graphs (HDF5 file)
GraphHDF5(pdb_path=pdb, pssm_path=pssm, biopython=False,
              graph_type='residue', outfile=database_test, nproc=8)

# In a benchmark mode, you can add the target values of your test set
# to comute the performance metrics
add_target(graph_path=database_test, target_name='bio_interface',
           target_list='bio_interfaces.txt')

# Make the prediction
model = NeuralNet(database_test, GINet, pretrained_model = "tclass_ybio_interface_b128_e50_lr0.001_26.pth.tar", target='bio_interface')
# if you have no target values (prediction mode), rmove the 'target' argument AND THE METRICS COMPUTATION LINES:
# model = NeuralNet(database_test, GINet, pretrained_model = "tclass_ybio_interface_b128_e50_lr0.001_26.pth.tar")
model.test(hdf5, hdf5='prediction_phy_non-phy.hdf5')

# Performance on the test set
test_metrics = model.get_metrics('test', threshold = 1.0)
print("accuracy:",test_metrics.accuracy)
print("specificity:",test_metrics.specificity)
print("sensitivity:",test_metrics.sensitivity)
print("precision:",test_metrics.precision)
print("FPR:",test_metrics.FPR)
print("FNR:",test_metrics.FNR)
