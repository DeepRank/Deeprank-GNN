import os
import numpy as np
from collections import OrderedDict, namedtuple
from iScore.graph import GenGraph
from deeprank.features.ResidueDensity import ResidueDensity
from deeprank.features.BSA import BSA
from graphprot.tools import BioWrappers
from time import time
from Graph import Graph
import networkx as nx


class ResidueGraph(Graph):

    def __init__(self,pdb=None,pssm={},contact_distance=8.5,internal_contact_distance = 8.5):
        super().__init__()

        self.type = 'residue'
        self.pdb = pdb
        self.name = os.path.splitext(os.path.basename(pdb))[0]
        self.pssm = pssm
        self.contact_distance = contact_distance
        self.internal_contact_distance = internal_contact_distance

        self.residue_charge = {'CYS':-0.64, 'HIS':-0.29, 'ASN':-1.22, 'GLN':-1.22, 'SER':-0.80, 'THR':-0.80, 'TYR':-0.80,
                               'TRP':-0.79, 'ALA':-0.37, 'PHE':-0.37, 'GLY':-0.37,'ILE':-0.37,'VAL':-0.37,'MET':-0.37,
                               'PRO':0.0,'LEU':-0.37, 'GLU':-1.37,'ASP':-1.37,'LYS':-0.36,'ARG':-1.65}


        self.residue_names = {'CYS':0, 'HIS':1, 'ASN':2, 'GLN':3, 'SER':4, 'THR':5, 'TYR':6, 'TRP':7,
                              'ALA':8, 'PHE':9, 'GLY':10,'ILE':11,'VAL':12,'MET':13,'PRO':14,'LEU':15,
                              'GLU':16,'ASP':17,'LYS':18,'ARG':20}


        self.residue_polarity = {'CYS':'polar','HIS':'polar','ASN':'polar','GLN':'polar','SER':'polar','THR':'polar','TYR':'polar','TRP':'polar',
                                 'ALA':'apolar','PHE':'apolar','GLY':'apolar','ILE':'apolar','VAL':'apolar','MET':'apolar','PRO':'apolar','LEU':'apolar',
                                 'GLU':'charged','ASP':'charged','LYS':'charged','ARG':'charged'}


        self.pssm_pos = {'CYS':4, 'HIS':8, 'ASN':2, 'GLN':5, 'SER':15, 'THR':16, 'TYR':18, 'TRP':17,
                              'ALA':0, 'PHE':13, 'GLY':7,'ILE':9,'VAL':19,'MET':12,'PRO':14,'LEU':10,
                              'GLU':6,'ASP':3,'LYS':11,'ARG':1}

        self.polarity_encoding = {'apolar':0,'polar':-1,'charged':1}
        self.edge_polarity_encoding,iencod = {}, 0
        for k1,v1 in self.polarity_encoding.items():
            for k2,v2 in self.polarity_encoding.items():
                key = tuple(np.sort([v1,v2]))
                if key not in self.edge_polarity_encoding:
                    self.edge_polarity_encoding[key] = iencod
                    iencod += 1

        # get the residue graph
        t0 = time()
        self.iScoreGraph = GenGraph(self.pdb,self.pssm,aligned=True,
                                    export=False,cutoff=contact_distance)
        self.iScoreGraph.construct_graph()
        print('iScore %f' %(time()-t0))

        # get the BSA of the residues
        t0 = time()
        self.BSA = BSA(self.pdb)
        self.BSA.get_structure()
        self.BSA.get_contact_residue_sasa(cutoff=contact_distance)
        print('BSA %f' %(time()-t0))

        #biopython data
        t0 = time()
        model = BioWrappers.get_bio_model(self.iScoreGraph.pdb)
        print('BioModel %f' %(time()-t0))

        t0 = time()
        self.ResDepth = BioWrappers.get_depth_res(model)
        print('BioDepth %f' %(time()-t0))

        t0 = time()
        self.HSE = BioWrappers.get_hse(model)
        print('BioHSE %f' %(time()-t0))

        # graph
        t0 = time()
        self.nx = nx.Graph()
        self.nx.edge_index = []

        # get nodes
        self.get_nodes()

        # get internal edges
        self.get_edges('internal')

        # get interface edges
        self.get_edges('interface')
        print('Graph %f' %(time()-t0))

    def get_nodes(self):

        for iRes,res in enumerate(self.iScoreGraph.nodes):

            node_key = self.iScoreGraph.pdb.get('chainID,resName,resSeq',chainID=res[0],resSeq=res[1])[0]
            node_key = tuple(map(str,node_key))
            self.nx.add_node(node_key)

            chainID = node_key[0]
            resName = node_key[1]
            resSeq = int(node_key[2])

            self.nx.nodes[node_key]['type'] = self.residue_names[resName]
            self.nx.nodes[node_key]['chain'] = {'A':0,'B':1}[res[0]]
            self.nx.nodes[node_key]['bsa'] = self.BSA.bsa_data[res]
            self.nx.nodes[node_key]['charge'] = self.residue_charge[resName]
            self.nx.nodes[node_key]['polarity'] = self.polarity_encoding[self.residue_polarity[resName]]
            self.nx.nodes[node_key]['pos'] = np.mean(self.iScoreGraph.pdb.get('x,y,z',chainID=res[0],resSeq=res[1]),0)
            self.nx.nodes[node_key]['pssm'] = [float(p) for p in self.iScoreGraph.aligned_pssm[res]]
            self.nx.nodes[node_key]['cons'] = float(self.iScoreGraph.aligned_pssm[res][self.pssm_pos[resName]])
            self.nx.nodes[node_key]['ic'] = float(self.iScoreGraph.aligned_ic[res])

            bio_key = (chainID,resSeq)
            self.nx.nodes[node_key]['depth'] = self.ResDepth[bio_key] if bio_key in self.ResDepth else 0
            self.nx.nodes[node_key]['hse'] = self.HSE[bio_key] if bio_key in self.HSE else 0

    def get_edges(self,edge_type):

        if edge_type == 'interface':
            index = self.iScoreGraph.edges

        elif edge_type == 'internal':
            index = self._get_internal_index()

        node_keys = list(self.nx.nodes)

        for i,j in index:

            node1, node2 = node_keys[i], node_keys[j]
            self.nx.add_edge(node1,node2)
            self.nx.edges[node1,node2]['type'] = bytes(edge_type,encoding='utf-8')
            self.nx.edges[node1,node2]['dist'] = self._get_edge_distance(node1,node2)
            self.nx.edges[node1,node2]['polarity'] = self._get_edge_polarity(node1,node2)
            self.nx.edge_index.append([node_keys.index(node1),node_keys.index(node2)])


    def _get_internal_index(self):

        indexA = np.unique(np.array(self.iScoreGraph.edges)[:,0])
        indexB = np.unique(np.array(self.iScoreGraph.edges)[:,1])
        nA, nB = len(indexA), len(indexB)

        nodeA = self.iScoreGraph.nodes[:nA]
        nodeB = self.iScoreGraph.nodes[nA:]

        edge_indexA,edge_weightA = [], []
        for i1 in range(nA-1):
            xyz1 = np.array(self.iScoreGraph.pdb.get('x,y,z',chainID=nodeA[i1][0],resSeq=nodeA[i1][1]))
            for i2 in range(i1+1,nA):
                xyz2 = np.array(self.iScoreGraph.pdb.get('x,y,z',chainID=nodeA[i2][0],resSeq=nodeA[i2][1]))
                dist = -2*np.dot(xyz1,xyz2.T) + np.sum(xyz1**2,axis=1)[:,None] + np.sum(xyz2**2,axis=1)
                if np.any(dist<self.internal_contact_distance):
                    edge_indexA.append([i1,i2])

        edge_indexB,edge_weightB = [], []
        for i1 in range(nB-1):
            xyz1 = np.array(self.iScoreGraph.pdb.get('x,y,z',chainID=nodeB[i1][0],resSeq=nodeB[i1][1]))
            for i2 in range(i1+1,nB):
                xyz2 = np.array(self.iScoreGraph.pdb.get('x,y,z',chainID=nodeB[i2][0],resSeq=nodeB[i2][1]))
                dist = -2*np.dot(xyz1,xyz2.T) + np.sum(xyz1**2,axis=1)[:,None] + np.sum(xyz2**2,axis=1)
                if np.any(dist<self.internal_contact_distance):
                    edge_indexB.append([i1+nA,i2+nA])


        return np.vstack((edge_indexA,edge_indexB))


    def _get_edge_polarity(self,node1,node2):

        v1 = self.nx.nodes[node1]['polarity']
        v2 = self.nx.nodes[node2]['polarity']

        key = tuple(np.sort([v1,v2]))
        return self.edge_polarity_encoding[key]


    def _get_edge_distance(self,node1,node2):

        pos1 = self.nx.nodes[node1]['pos']
        pos2 = self.nx.nodes[node2]['pos']
        return np.linalg.norm(pos1-pos2)




if __name__ == "__main__":

    import h5py
    ref = 'data/pdb/1ATN/1ATN.pdb'
    pdb = 'data/pdb/1ATN/1ATN_143w.pdb'
    pssm = {'A':'./data/pssm/1ATN/1ATN.A.pdb.pssm','B':'./data/pssm/1ATN/1ATN.B.pdb.pssm'}
    graph = ResidueGraph(pdb,pssm)
    graph.get_score(ref)

    f5 = h5py.File('test.hdf5','w')
    graph.nx2h5(f5)
    f5.close()


