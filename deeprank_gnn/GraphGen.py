import os
import sys
import h5py
from tqdm import tqdm
import time

from .ResidueGraph import ResidueGraph
from .Graph import Graph


class GraphHDF5(object):

    def __init__(self, pdb_path, ref_path=None, graph_type='residue', pssm_path=None,
                 select=None, outfile='graph.hdf5', biopython=False):
        """Master class from which graphs are computed 

        Args:
            pdb_path (str): path to the docking models
            ref_path (str, optional): path to the reference model. Defaults to None.
            graph_type (str, optional): Defaults to 'residue'.
            pssm_path ([type], optional): path to the pssm file. Defaults to None.
            select (str, optional): filter files that starts with 'input'. Defaults to None.
            outfile (str, optional): Defaults to 'graph.hdf5'.
        """
        # get the list of PDB names
        pdbs = list(filter(lambda x: x.endswith(
            '.pdb'), os.listdir(pdb_path)))
        if select is not None:
            pdbs = list(filter(lambda x: x.startswith(select), pdbs))

        f5 = h5py.File(outfile, 'w')

        desc = '{:25s}'.format('   Create HDF5')
        data_tqdm = tqdm(pdbs, desc=desc, file=sys.stdout)

        for name in data_tqdm:

            # pdb name
            pdbfile = os.path.join(pdb_path, name)

            # mol name and base name
            mol_name = os.path.splitext(name)[0]
            base_name = mol_name.split('_')[0]

            if graph_type == 'residue':
                # get the pssm file for pdb
                if pssm_path is not None: 
                    pssm = self._get_pssm(pssm_path, mol_name, base_name)
                else pssm = None
                # generate a graph 
                graph = ResidueGraph(pdb=pdbfile, pssm=pssm, biopython=biopython)

            # get the score
            if ref_path is not None:
                ref = os.path.join(ref_path, base_name+'.pdb')
                graph.get_score(ref)

            # export the graph in hdf5 format
            try:
                graph.nx2h5(f5)
            except:
                print('WARNING: No graph generated for {}'.format(name))

        f5.close()

    @staticmethod
    def _get_pssm(pssm_path, mol_name, base_name):
        """Gets the pssm file for a given model

        Args:
            pssm_path (str): path to the pssm file
            mol_name (str): molecule name (XXXX_A.pdb -> XXXX_A)
            base_name (str): short molecule name (XXXX_A.pdb -> XXXX)

        Returns:
            pssm files for chain A and for chain B 
            pssm = {'A': pssmA, 'B': pssmB}
        """
        pssmA = os.path.join(pssm_path, base_name+'.A.pdb.pssm')
        pssmB = os.path.join(pssm_path, base_name+'.B.pdb.pssm')

        # check if the pssms exists
        if os.path.isfile(pssmA) and os.path.isfile(pssmB):
            pssm = {'A': pssmA, 'B': pssmB}
        else:
            pssmA = os.path.join(pssm_path, base_name+'.A.pssm')
            pssmB = os.path.join(pssm_path, base_name+'.B.pssm')
            if os.path.isfile(pssmA) and os.path.isfile(pssmB):
                pssm = {'A': pssmA, 'B': pssmB}
            else:
                raise FileNotFoundError(
                    'PSSM file for ' + mol_name + ' not found')
        return pssm

if __name__ == '__main__':

    pdb_path = './data/pdb/1ATN/'
    pssm_path = './data/pssm/1ATN/'
    ref = './data/ref/1ATN/'

    GraphHDF5(pdb_path=pdb_path, ref_path=ref, pssm_path=pssm_path,
              graph_type='residue', outfile='1AK4_residue.hdf5')
