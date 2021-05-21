
.. _Graph Generation tools:

Creating Graphs
=====================================

.. warning::
  The graph generation requires an ensemble of PDB files containing two chains: chain **A** and chain **B**.
  
  You can provide PSSM matrices to compute evolutionary conservation node features. Some pre-calculated PSSM matrices can be downloaded from http://3dcons.cnb.csic.es/.
  A ``3dcons_to_deeprank_pssm.py`` converter can be found in the ``tool`` folder to convert the 3dcons PSSM format into the Deeprank-GNN PSSM format. **Make sure the sequence numbering matches the PDB residues numbering.**

Training mode 
-------------------------------------

In a training mode, you are required to provide the path to the reference structures in the graph generation step. Knowing the reference structure, the following target values based on CAPRI quality criteria [REFE] will be automatically computed and assigned to the graphs : 

- **irmsd**: interface RMSD (RMSD between the superimposed interface residues)

- **lrmsd**: ligand RMSD (RMSD between chains B given that chains A are superimposed)

- **fnat**: fraction of native contacts

- **dockQ**: see Basu et al., "DockQ: A Quality Measure for Protein-Protein Docking Models", PLOS ONE, 2016

- **bin_class**: binary classification (0: ``irmsd >= 4 A``, 1: ``RMSD < 4A``)

- **capri_classes**: 1: ``RMSD < 1A``, 2: ``RMSD < 2A``, 3: ``RMSD < 4A``, 4: ``RMSD < 6A``, 0: ``RMSD >= 6A``


>>> from deeprank_gnn.GraphGenMP import GraphHDF5
>>>
>>> pdb_path = './data/pdb/1ATN/'
>>> pssm_path = './data/pssm/1ATN/'
>>> ref = './data/ref/1ATN/'
>>>
>>> GraphHDF5(pdb_path=pdb_path, ref_path=ref, pssm_path=pssm_path,
>>>          graph_type='residue', outfile='1ATN_residue.hdf5', nproc=4)

.. note::  
  The different input files must respect the following nomenclature:
  
   - PDB files: ``1ATN_xxx.pdb`` (xxx may be replaced by anything)
   - PSSM files: ``1ATN.A.pdb.pssm 1ATN.B.pdb.pssm`` or ``1ATN.A.pssm 1ATN.B.pssm``
   - Reference PDB files: ``1ATN.pdb``
   

Prediction mode
-------------------------------------

In a prediction mode, you may use a pre-trained model and run it on graphs for which no experimental structure is available. 
No targets will be computed.

>>> from deeprank_gnn.GraphGenMP import GraphHDF5
>>>
>>> pdb_path = './data/pdb/1ATN/'
>>> pssm_path = './data/pssm/1ATN/'
>>>
>>> GraphHDF5(pdb_path=pdb_path, pssm_path=pssm_path,
>>>          graph_type='residue', outfile='1ATN_residue.2.hdf5', nproc=4)

Add your own target values
-------------------------------------

The automatically computed target values are docking related, which may not match your research requirement.

You may instead generate the PPI graphs and add your own target values.

>>> from deeprank_gnn.GraphGen import GraphHDF5
>>> from deeprank_gnn.CustomizeGraph import CustomizeGraph
>>>
>>> pdb_path = './data/pdb/1ATN/'
>>> pssm_path = './data/pssm/1ATN/'
>>>
>>> GraphHDF5(pdb_path=pdb_path, pssm_path=pssm_path,
>>>          graph_type='residue', outfile='1ATN_residue.hdf5', nproc=4)
>>>
>>> CustomizeGraph.add_target(graph_path='.', target_name='new_target',
>>>                           target_list=list_of_target_values.txt)

.. note::
  The list of target values should respect the following format:
  
  ``1ATN_xxx-1 0``
  
  ``1ATN_xxx-2 1``
  
  ``1ATN_xxx-3 0``
  
  ``1ATN_xxx-4 0``
  
  if your use other separators (eg. ``,``, ``;``, ``tab``) use the ``sep`` argument:
  
  >>> CustomizeGraph.add_target(pdb_path=pdb_path, target_name='new_target', 
  >>>                           target_list=list_of_target_values.txt, sep=',')
  
