
.. _Creating Graphs:

Creating Graphs
=====================================

.. warning::
  The graph generation requires an ensemble of PDB files containing two chains: chain **A** and chain **B**.
  
  You can provide PSSM matrices to compute evolutionary conservation node features. Some pre-calculated PSSM matrices can be downloaded from http://3dcons.cnb.csic.es/.
  A ``3dcons_to_deeprank_pssm.py`` converter can be found in the ``tool`` folder to convert the 3dcons PSSM format into the Deeprank-GNN PSSM format. **Make sure the sequence numbering matches the PDB residues numbering.**
  
  
 
  By default, the following features are assigned to each node of the graph :
  
- **pos**: xyz coordinates

- **chain**: chain ID

- **charge**: residue charge

- **polarity**: apolar/polar/neg_charged/pos_charged (one hot encoded)

- **bsa**: buried surface are

- **pssm**: pssm score for each residues

- **cons**: pssm score of the residue

- **ic**: information content of the PSSM (~Shannon entropy)

- **type**: residue type (one hot encoded)

The following one are optional, and require more computation time (~x10) :

- **depth**: average atom depth of the atoms in a residue (distance to the surface)

- **hse**: half sphere exposure

Generate your graphs 
-------------------------------------

Note that the pssm information is used to compute the **pssm**, **cons** and **ic** node features and is optional.

>>> from deeprank_gnn.GraphGenMP import GraphHDF5
>>>
>>> pdb_path = './data/pdb/1ATN/'
>>> pssm_path = './data/pssm/1ATN/'
>>>
>>> GraphHDF5(pdb_path=pdb_path, pssm_path=pssm_path,
>>>          graph_type='residue', outfile='1ATN_residue.hdf5', nproc=4)

Add your target values
-------------------------------------

Use the CustomizeGraph class to add target values to the graphs. 

If you are benchmarking docking models, go to the **next section**.

>>> from deeprank_gnn.GraphGen import GraphHDF5
>>> import deeprank_gnn.CustomizeGraph as CustomizeGraph
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
  
  ``model_name_1 0``
  
  ``model_name_2 1``
  
  ``model_name_3 0``
  
  ``model_name_4 0``
  
  if your use other separators (eg. ``,``, ``;``, ``tab``) use the ``sep`` argument:
  
  >>> CustomizeGraph.add_target(graph_path=graph_path, target_name='new_target', 
  >>>                           target_list='list_of_target_values.txt', sep=',')
  
  
Docking benchmark mode 
-------------------------------------

In a docking benchmark mode, you cano provide the path to the reference structures in the graph generation step. Knowing the reference structure, the following target values will be automatically computed, based on CAPRI quality criteria [1]_,  and assigned to the graphs : 

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
   


.. [1] 
  Lensink MF, Méndez R, Wodak SJ, Docking and scoring protein complexes: CAPRI 3rd Edition. Proteins. 2007
