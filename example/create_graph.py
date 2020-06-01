from graphprot.GraphGenMP import GraphHDF5

pdb_path = './pdb/1ATN/'
pssm_path = './pssm/1ATN/'
ref = './ref/1ATN/'

GraphHDF5(pdb_path=pdb_path, ref_path=ref, pssm_path=pssm_path,
          graph_type='residue', outfile='1ATN_residue.hdf5',
          nproc=2, tmpdir='./tmpdir')
