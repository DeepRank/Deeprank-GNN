# Run DeepRank-GNN pre-trained model on your own data

We herein provide the pretrained model `fold6_treg_yfnat_b128_e20_lr0.001_4.pt` from the DeepRank-GNN paper (doi: 10.1101/2021.12.08.471762).

An example of code to run DeepRank-GNN on new data with the pre-trained model is provided (*test.py*).

## Usage: 
`python test.py `


You can also check Deeprank-GNN documentation: [`use-deeprank-gnn-paper-s-pretrained-model`](https://deeprank-gnn.readthedocs.io/en/latest/tutorial.train_model.html#use-deeprank-gnn-paper-s-pretrained-model)

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
pdb_path = '../tests/data/pdb/1ATN/' 
#path to the pssm files
pssm_path = '../tests/data/pssm/1ATN/'

GraphHDF5(pdb_path=pdb_path, pssm_path=pssm_path,
        graph_type='residue', outfile='1ATN_residue.hdf5', nproc=4)
```

### Prediction section
```
pretrained_model = 'fold6_treg_yfnat_b128_e20_lr0.001_4.pt'
gnn = GINet

#path to the graph(s)
database_test = glob.glob('./*.hdf5')

start_time = time.time()
model = NeuralNet(database_test, gnn, pretrained_model = pretrained_model)    
model.test(threshold=None)
end_time = time.time()

print ('Elapsed time: {end_time-start_time}')
```

DeepRank-GNN will generate a HDF5 file with the prediction. We provite a hdf5 to csv converter to easily read it :
`python Deeprank-GNN/deeprank_gnn/tools/hdf5_to_csv.py your.hdf5`

Further information can be found in the DeepRank-GNN online [documentation](https://deeprank-gnn.readthedocs.io/en/latest/index.html)
