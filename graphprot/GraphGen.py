import os, sys
import h5py
from AtomicGraph import AtomicGraph
from ResidueGraph import ResidueGraph
from Graph import Graph
from tqdm import tqdm
import time

class GraphHDF5(object):

    def __init__(self,pdb_path,ref_path,graph_type='atomic',pssm_path=None,
                select=None,outfile='graph.hdf5',line=False):

        # get the list of PDB names
        pdbs = list(filter(lambda x: x.endswith('.pdb'),os.listdir(pdb_path)))
        if select is not None:
            pdbs = list(filter(lambda x: x.startswith(select),pdbs))


        f5 = h5py.File(outfile,'w')
        if line:
            f5line = h5py.File('line'+outfile,'w')


        desc = '{:25s}'.format('   Create HDF5')
        data_tqdm = tqdm(pdbs,desc=desc,file=sys.stdout)
        #data_tqdm = pdbs[:1]

        for name in data_tqdm:

            #print('Creating graph of PDB %s' %name)

            # pdb name
            pdbfile = os.path.join(pdb_path,name)

            # mol name and base name
            mol_name = os.path.splitext(name)[0]
            base_name = mol_name.split('_')[0]

            if graph_type == 'atomic':
                #t0 = time.time()
                graph = AtomicGraph(pdb=pdbfile)
                #print('Create Graph %f' %(time.time()-t0))

            elif graph_type == 'residue':
                pssm = self._get_pssm(pssm_path,mol_name,base_name)
                graph = ResidueGraph(pdb=pdbfile,pssm=pssm)

            # get the score
            ref = os.path.join(ref_path,base_name+'.pdb')
            #t0 = time.time()
            graph.score(pdbfile,ref)
            #print('Score  Graph %f' %(time.time()-t0))

            #export
            #t0 = time.time()
            graph.export_hdf5(f5)
            #print('Export Graph %f' %(time.time()-t0))

            # get the line graph
            if line:
                #t0 = time.time()
                lgraph = Graph.get_line_graph(graph)
                lgraph.export_hdf5(f5line)
                #print('Line Graph %f' %(time.time()-t0))

        f5.close()
        if line:
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

    GraphHDF5(pdb_path=pdb_path,ref_path=ref,pssm_path=pssm_path,
              graph_type='atomic',outfile='graph_atomic.hdf5')


