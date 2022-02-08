.. _Training a module:

Training a module
=============================================


1. Select node and edge features
---------------------------------------------

1.1. Edge feature:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **dist**: distance between nodes

>>> edge_feature=['dist']


.. note::  
  **External edges** connect 2 residues of chain A and B if they have at least 1 pairwise atomic distance **< 8.5 A** (Used for to define neighbors)
  
  **Internal edges** connect 2 residues within a chain if they have at least 1 pairwise atomic distance **< 3 A** (Used to cluster nodes)



1.2. Node features:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **pos**: xyz coordinates

- **chain**: chain ID

- **charge**: residue charge

- **polarity**: apolar/polar/neg_charged/pos_charged (one hot encoded)

- **bsa**: buried surface are

- **pssm**: pssm score for each residues

- **cons**: pssm score of the residue

- **ic**: information content of the PSSM (~Shannon entropy)

- **type**: residue type (one hot encoded)

- **depth** (opt. in graph. gen. step): average atom depth of the atoms in a residue (distance to the surface)

- **hse** (opt. in graph. gen. step): half sphere exposure



>>> node_feature=['type', 'polarity', 'bsa',
>>>               'ic', 'pssm']


2. Select the target (benchmarking mode)
---------------------------------------------

When using Deeprank-GNN in a benchmarking mode, you must specify your target (often referred to as Y).
The target values are pre-calculated during the Graph generation step **if a reference structure is provided**.

**Pre-calculated targets:** 

- **irmsd**: interface RMSD (RMSD between the superimposed interface residues)

- **lrmsd**: ligand RMSD (RMSD between chains B given that chains A are superimposed)

- **fnat**: fraction of native contacts

- **dockQ**: see Basu et al., "DockQ: A Quality Measure for Protein-Protein Docking Models", PLOS ONE, 2016

- **bin_class**: binary classification (0: ``irmsd >= 4 A``, 1: ``RMSD < 4A``)

- **capri_classes**: 1: ``RMSD < 1A``, 2: ``RMSD < 2A``, 3: ``RMSD < 4A``, 4: ``RMSD < 6A``, 0: ``RMSD >= 6A``

.. note::  
 In classification mode (i.e. task="class") you must provide the list of target classes to the NeuralNet (e.g. classes=[1,2,3,4])


>>> target='irmsd'

3. Select hyperparameters
---------------------------------------------

- Regression ('reg') of classification ('class') mode

>>> task='reg' 

- Batch size

>>> batch_size=64

- Shuffle the training dataset

>>> shuffle=True

- Learning rate:

>>> lr=0.001

4. Load the network
---------------------------------------------

This step requires pre-calculated graphs in hdf5 format. 

The user may :

- option 1: input a unique dataset and chose to automatically split it into a training set and an evaluation set

- option 2: input distinct training/evaluation/test sets

4.1. Option 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> from deeprank_gnn.NeuralNet import NeuralNet
>>> from deeprank_gnn.ginet import GINet
>>>
>>> database = './1ATN_residue.hdf5'
>>>
>>> model = NeuralNet(database, GINet,
>>>                node_feature=node_feature,
>>>                edge_feature=edge_feature,
>>>                target=target,
>>>                task=task, 
>>>                lr=lr,
>>>                batch_size=batch_size,
>>>                shuffle=shuffle,
>>>                percent=[0.8, 0.2])
>>>

.. note::  
 The *percent* argument is required to split the input dataset into a training set and a test set. Using ``percent=[0.8, 0.2]``, 80% of the input dataset will constitute the  training set, 20% will constitute the evaluation set. 

4.2. Option 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> from deeprank_gnn.NeuralNet import NeuralNet
>>> from deeprank_gnn.ginet import GINet
>>> import glob 
>>>
>>> # load train dataset
>>> database_train = glob.glob('./hdf5/train*.hdf5')
>>> # load validation dataset
>>> database_eval = glob.glob('./hdf5/eval*.hdf5')
>>> # load test dataset
>>> database_test = glob.glob('./hdf5/test*.hdf5')
>>> 
>>> model = NeuralNet(database_train, GINet,
>>>                node_feature=node_feature,
>>>                edge_feature=edge_attr,
>>>                target=target,
>>>                task=task, 
>>>                lr=lr,
>>>                batch_size=batch_size,
>>>                shuffle=shuffle,
>>>                database_eval = database_eval)

5. Train the model 
---------------------------------------------

- example 1:

train the network, perform 50 epochs

>>> model.train(nepoch=50, validate=False)

- example 2:

train the model, evaluate the model at each epoch, save the best model (i.e. the model with the lowest loss), and write all predictions to ``output.hdf5``

>>> model.train(nepoch=50, validate=True, save_model='best', hdf5='output.hdf5')

.. warning::
 The ``last`` model is saved by default.
 
 When setting ``save_model='best'``, a model that is associated with a lower loss than those generated in the previous epochs will be saved. By default, the epoch number is included in the output name not to write over intermediate models.

6. Analysis
---------------------------------------------

6.1. Plot the loss evolution over the epochs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> model.plot_loss(name='plot_loss')

6.2 Analyse the performance in benchmarking conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following analysis only apply if a reference structure was provided during the graph generation step.

6.2.1. **Plot accuracy evolution**

>>> model.plot_acc(name='plot_accuracy')

6.2.2. **Plot hitrate**

A threshold value is required to binarise the target value

>>> model.plot_hit_rate(data='eval', threshold=4.0, mode='percentage', name='hitrate_eval')

6.2.3. **Get various metrics**

The following metrics can be easily computed: 

**Classification metrics:**

- **sensitivity**: Sensitivity, hit rate, recall, or true positive rate

- **specificity**: Specificity or true negative rate

- **precision**: Precision or positive predictive value

- **NPV**: Negative predictive value

- **FPR**: Fall out or false positive rate

- **FNR**: False negative rate

- **FDR**: False discovery rate

- **accuracy**: Accuracy

- **auc()**: AUC

- **hitrate()**: Hit rate

**Regression metrics:**

- **explained_variance**: Explained variance regression score function

- **max_error**: Max_error metric calculates the maximum residual error

- **mean_abolute_error**: Mean absolute error regression loss

- **mean_squared_error**: Mean squared error regression loss

- **root_mean_squared_error**: Root mean squared error regression loss

- **mean_squared_log_error**: Mean squared logarithmic error regression loss

- **median_squared_log_error**: Median absolute error regression loss

- **r2_score**: R^2 (coefficient of determination) regression score function

.. note::  
  All classification metrics can be calculated on continuous targets as soon as a threshold is provided to binarise the data.

>>> train_metrics = model.get_metrics('train', threshold = 4.0)
>>> print('training set - accuracy:', train_metrics.accuracy)
>>> print('training set - sensitivity:', train_metrics.sensitivity)
>>> 
>>> eval_metrics = model.get_metrics('eval', threshold = 4.0)
>>> print('evaluation set - accuracy:', eval_metrics.accuracy)
>>> print('evaluation set - sensitivity:', eval_metrics.sensitivity)

7. Save the model/network
---------------------------------------------

>>> model.save_model("model_backup.pth.tar")

8. Test the model on an external dataset
---------------------------------------------

8.1. On a loaded model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> model.test(database_test, threshold=4.0)

8.2. On a pre-trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> from deeprank_gnn.NeuralNet import NeuralNet
>>> from deeprank_gnn.ginet import GINet
>>>  
>>> database_test = './1ATN_residue.hdf5'
>>>  
>>> model = NeuralNet(database_test, GINet, pretrained_model = "model_backup.pth.tar")
>>> model.test(database_test)
>>>  
>>> test_metrics = model.get_metrics('test', threshold = 4.0)
>>> print(test_metrics.accuracy)

In short 
=============================================

>>> from deeprank_gnn.NeuralNet import NeuralNet
>>> from deeprank_gnn.ginet import GINet
>>>
>>> database = './1ATN_residue.hdf5'
>>>
>>> edge_feature=['dist']
>>> node_feature=['type', 'polarity', 'bsa',
>>>               'depth', 'hse', 'ic', 'pssm']
>>> target='irmsd'
>>> task='reg' 
>>> batch_size=64
>>> shuffle=True
>>> lr=0.001
>>>
>>> model = NeuralNet(database, GINet,
>>>                node_feature=node_feature,
>>>                edge_feature=edge_feature,
>>>                target=target,
>>>                index=None,
>>>                task=task, 
>>>                lr=lr,
>>>                batch_size=batch_size,
>>>                shuffle=shuffle,
>>>                percent=[0.8, 0.2])
>>>
>>> model.train(nepoch=50, validate=True, save_model='best', hdf5='output.hdf5')
>>> model.plot_loss(name='plot_loss')
>>> 
>>> train_metrics = model.get_metrics('train', threshold = 4.0)
>>> print('training set - accuracy:', train_metrics.accuracy)
>>> print('training set - sensitivity:', train_metrics.sensitivity)
>>> 
>>> eval_metrics = model.get_metrics('eval', threshold = 4.0)
>>> print('evaluation set - accuracy:', eval_metrics.accuracy)
>>> print('evaluation set - sensitivity:', eval_metrics.sensitivity)
>>> 
>>> model.save_model("model_backup.pth.tar")
>>> #model.test(database_test, threshold=4.0)

Using default settings 

>>> from deeprank_gnn.NeuralNet import NeuralNet
>>> from deeprank_gnn.ginet import GINet
>>>
>>> database = glob.glob('./hdf5/*_train.hdf5')
>>> dataset_test = glob.glob('./hdf5/*_test.hdf5')
>>>
>>> target='irmsd'
>>>
>>> model = NeuralNet(database, GINet,
>>>                target=target,
>>>                percent=[0.8, 0.2])
>>>
>>> model.train(nepoch=50, validate=True, save_model='best', hdf5='output.hdf5')
>>> model.plot_loss(name='plot_loss')
>>> 
>>> train_metrics = model.get_metrics('train', threshold = 4.0)
>>> print('training set - accuracy:', train_metrics.accuracy)
>>> print('training set - sensitivity:', train_metrics.sensitivity)
>>> 
>>> eval_metrics = model.get_metrics('eval', threshold = 4.0)
>>> print('evaluation set - accuracy:', eval_metrics.accuracy)
>>> print('evaluation set - sensitivity:', eval_metrics.sensitivity)
>>> 
>>> model.save_model("model_backup.pth.tar")
>>> model.test(database_test, threshold=4.0)

Use DeepRank-GNN paper's pretrained model
=============================================

**See**: M. Réau, N. Renaud, L. C. Xue, A. M. J. J. Bonvin, "DeepRank-GNN: A Graph Neural Network Framework to Learn Patterns in Protein-Protein Interfaces", bioRxiv 2021.12.08.471762; doi: https://doi.org/10.1101/2021.12.08.471762

You can get the `pre-trained model <https://github.com/DeepRank/Deeprank-GNN/tree/master/paper_pretrained_models/>`_ from DeepRank-GNN `github repository <https://github.com/DeepRank/Deeprank-GNN/>`_

>>> import glob 
>>> import sys 
>>> import time
>>> import datetime 
>>> import numpy as np
>>> 
>>> from deeprank_gnn.GraphGenMP import GraphHDF5
>>> from deeprank_gnn.NeuralNet import NeuralNet
>>> from deeprank_gnn.ginet import GINet
>>>
>>> ### Graph generation section
>>> pdb_path = '../tests/data/pdb/1ATN/'
>>> pssm_path = '../tests/data/pssm/1ATN/'
>>>
>>> GraphHDF5(pdb_path=pdb_path, pssm_path=pssm_path,
>>>         graph_type='residue', outfile='1ATN_residue.hdf5', nproc=4)
>>> 
>>> ### Prediction section
>>> gnn = GINet
>>> pretrained_model = 'fold6_treg_yfnat_b128_e20_lr0.001_4.pt'
>>> database_test = glob.glob('1ATN_residue.hdf5')
>>> 
>>> start_time = time.time()
>>> model = NeuralNet(database_test, gnn, pretrained_model = pretrained_model)    
>>> model.test(threshold=None)
>>> end_time = time.time()
>>> print ('Elapsed time: {end_time-start_time}')
>>> 
>>> ### The output is automatically stored in **test_data.hdf5** 
         
.. note::  
 For storage convenience, all predictions are stored in a HDF5 file. A converter from HDF5 to csv is provided in the **tools** directory
