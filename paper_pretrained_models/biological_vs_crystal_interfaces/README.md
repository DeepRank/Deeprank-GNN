# Run DeepRank-GNN pre-trained model on your own data

We herein provide the pretrained model `tclass_ybio_interface_b128_e50_lr0.001_26.pth.tar` from the DeepRank-GNN paper (doi: 10.1101/2021.12.08.471762).

An example of code to run DeepRank-GNN on new data with the pre-trained model is provided (*test.py*).

## Usage: 
`python test.py `


The data used to train the model are available on [SBGrid](https://data.sbgrid.org/dataset/843/)

## Please cite

M. Réau, N. Renaud, L. C. Xue, A. M. J. J. Bonvin, DeepRank-GNN: A Graph Neural Network Framework to Learn Patterns in Protein-Protein Interfaces
bioRxiv 2021.12.08.471762; doi: https://doi.org/10.1101/2021.12.08.471762

## Details of the code

```
import glob 
import sys 
import time
import datetime 
import numpy as np

from deeprank_gnn.GraphGenMP import GraphHDF5
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import GINet
```
### Graph generation section
```
#path to the docking models in pdb format
pdb_path = '../DC/pdb/' 
#path to the pssm files
pssm_path = '../DC/pssm/'

#path to the graphs (hdf5) file 
database_test = 'biological_vs_crystal.hdf5'

#generation of the graphs (HDF5 file)
GraphHDF5(pdb_path=pdb, pssm_path=pssm, biopython=False,
              graph_type='residue', outfile=database_test, nproc=8)
```
In a benchmark mode, you can add the target values of your test set to compute the performance metrics
More details are provided in DeepRank-GNN's [online documentation](https://deeprank-gnn.readthedocs.io/en/latest/tutorial.generate_graph.html#add-your-target-values)
```
add_target(graph_path=database_test, target_name='bio_interface',
           target_list='bio_interfaces.txt')

```
### Prediction section
```
pretrained_model = 'tclass_ybio_interface_b128_e50_lr0.001_26.pth.tar'
gnn = GINet

start_time = time.time()
model = NeuralNet(database_test, gnn, pretrained_model = pretrained_model", target='bio_interface')

# if you have no target values (prediction mode), remove the 'target' argument AND THE METRICS COMPUTATION LINES:
# model = NeuralNet(database_test, GINet, pretrained_model = "tclass_ybio_interface_b128_e50_lr0.001_26.pth.tar")
model.test(hdf5, hdf5='prediction_phy_non-phy.hdf5')
end_time = time.time()

print ('Elapsed time: {end_time-start_time}')
```
# Evaluation of the performance 
```
test_metrics = model.get_metrics('test', threshold = 1.0)
print("accuracy:",test_metrics.accuracy)
print("specificity:",test_metrics.specificity)
print("sensitivity:",test_metrics.sensitivity)
print("precision:",test_metrics.precision)
print("FPR:",test_metrics.FPR)
print("FNR:",test_metrics.FNR)
```

DeepRank-GNN will generate a HDF5 file with the prediction. We provite a hdf5 to csv converter to easily read it :
`python Deeprank-GNN/deeprank_gnn/tools/hdf5_to_csv.py your.hdf5`

Further information can be found in the DeepRank-GNN online [documentation](https://deeprank-gnn.readthedocs.io/en/latest/index.html)
