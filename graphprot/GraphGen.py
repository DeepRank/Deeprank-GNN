import os
import h5py
from AtomicGraph import AtomicGraph
from ResidueGraph import ResidueGraph
from Graph import Graph

class GraphHDF5(object):

    def __init__(self,pdb_path,ref_path,graph_type='atomic',pssm_path=None,select=None,outfile='graph.hdf5'):

        # get the list of PDB names
        pdbs = list(filter(lambda x: x.endswith('.pdb'),os.listdir(pdb_path)))
        if select is not None:
            pdbs = list(filter(lambda x: x.startswith(select),pdbs))


        f5 = h5py.File(outfile,'w')
        f5line = h5py.File('line'+outfile,'w')

        for name in pdbs:

            print('Creating graph of PDB %s' %name)

            # pdb name
            pdbfile = os.path.join(pdb_path,name)

            # mol name and base name
            mol_name = os.path.splitext(name)[0]
            base_name = mol_name.split('_')[0]

            if graph_type == 'atomic':
                graph = AtomicGraph(pdb=pdbfile)

            elif graph_type == 'residue':
                pssm = self._get_pssm(pssm_path,mol_name,base_name)
                graph = ResidueGraph(pdb=pdbfile,pssm=pssm)

            # get the score
            ref = os.path.join(ref_path,base_name+'.pdb')
            graph.score(pdbfile,ref)

            #export
            graph.export_hdf5(f5)

            # get the line graph
            lgraph = Graph.get_line_graph(graph)
            lgraph.export_hdf5(f5line)

        f5.close()
        f5line.close()

    @staticmethod
    def _get_pssm(pssm_path,mol_name,base_name):

        pssmA = os.path.join(pssm_path,mol_name+'.A.pdb.pssm')
        pssmB = os.path.join(pssm_path,mol_name+'.B.pdb.pssm')

        # check if the pssms exists
        if os.path.isfile(pssmA) and os.path.isfile(pssmB):
            pssm = {'A':pssmA,'B':pssmB}
        else:
            pssmA = os.path.join(pssm_path,base_name+'.A.pdb.pssm')
            pssmB = os.path.join(pssm_path,base_name+'.B.pdb.pssm')
            if os.path.isfile(pssmA) and os.path.isfile(pssmB):
                pssm = {'A':pssmA,'B':pssmB}
            else:
                raise FileNotFoundError('PSSM file for ' + mol_name + ' not found')
        return pssm

if __name__ == '__main__':

    pdb_path = './data/pdb'
    pssm_path = './data/pssm'
    ref = './data/ref'

    GraphHDF5(pdb_path=pdb_path,ref_path=ref,pssm_path=pssm_path,graph_type='residue',outfile='graph_residue.hdf5')


