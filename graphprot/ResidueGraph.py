import os
import numpy as np
from collections import OrderedDict, namedtuple
from iScore.graph import GenGraph
from deeprank.features.ResidueDensity import ResidueDensity
from deeprank.features.BSA import BSA
from Graph import Graph
import networkx as nx


class ResidueGraph(Graph):

    def __init__(self,pdb=None,pssm={},contact_distance=8.5,internal_contact_distance = 8.5,
        node_feature_str = 'type, bsa, pssm',
        edge_feature_str = 'dist, polarity'):
        super().__init__()

        self.type='residue'
        self.pdb = pdb
        self.name = os.path.splitext(os.path.basename(pdb))[0]
        self.pssm=pssm
        self.contact_distance = contact_distance
        self.internal_contact_distance = internal_contact_distance
        self.node_feature_str = [name.strip() for name in node_feature_str.split(',')]
        self.edge_feature_str = [name.strip() for name in edge_feature_str.split(',')]

        # get the resisude graph
        self.iScoreGraph = GenGraph(self.pdb,self.pssm,aligned=True,
                                    export=False,cutoff=contact_distance)
        self.iScoreGraph.construct_graph()

        # get the BSA of the residues
        self.BSA = BSA(self.pdb)
        self.BSA.get_structure()
        self.BSA.get_contact_residue_sasa(cutoff=contact_distance)

        # get the residue density
        self._get_edge_polarity()

        # get the edge distance
        self._get_edge_distance()

        # data
        self._get_nodes()
        self._get_edges()
        self._get_internal_edges()


    def _get_edge_polarity(self):

        residue_types = {'CYS':'polar','HIS':'polar','ASN':'polar','GLN':'polar','SER':'polar','THR':'polar','TYR':'polar','TRP':'polar',
                         'ALA':'apolar','PHE':'apolar','GLY':'apolar','ILE':'apolar','VAL':'apolar','MET':'apolar','PRO':'apolar','LEU':'apolar',
                         'GLU':'charged','ASP':'charged','LYS':'charged','ARG':'charged'}

        encoding = {}
        iencod = 0
        types = ['polar','apolar','charged']
        self.edge_polarity = []
        for t1 in types:
            for t2 in types:
                encoding[t1+t2] = iencod
                iencod += 1

        for i,j in self.iScoreGraph.edges:

            resi = self.iScoreGraph.nodes[i][2]
            resj = self.iScoreGraph.nodes[j][2]

            ti = residue_types[resi]
            tj = residue_types[resj]

            self.edge_polarity.append(encoding[ti+tj])


    def _get_edge_distance(self,center='mean'):

        self.pos = []

        for iRes,res in enumerate(self.iScoreGraph.nodes):

            if center == 'CA':
                self.pos.append(self.iScoreGraph.pdb.get('x,y,z',chainID=res[0],resSeq=res[1],name='CA'))
            elif center == 'mean':
                pos = np.array(self.iScoreGraph.pdb.get('x,y,z',chainID=res[0],resSeq=res[1]))
                self.pos.append(np.mean(pos,0))

        self.edge_distance = []
        self.pos = np.array(self.pos)
        for i,j in self.iScoreGraph.edges:
            self.edge_distance.append(np.linalg.norm(self.pos[i,:]-self.pos[j,:]))

    def _get_nodes(self):

        self.residue_encoding = {'CYS':0,'HIS':1,'ASN':2,'GLN':3,'SER':4,'THR':5,'TYR':6,'TRP':7,
                 'ALA':8,'PHE':9,'GLY':10,'ILE':11,'VAL':12,'MET':13,'PRO':14,'LEU':15,
                 'GLU':16,'ASP':17,'LYS':18,'ARG':20}


        self.node = []
        self.node_info = []
        dict_feat = {}

        for iRes,res in enumerate(self.iScoreGraph.nodes):

            dict_feat['pssm'] = [float(p) for p in self.iScoreGraph.aligned_pssm[res]]
            dict_feat['bsa'] = self.BSA.bsa_data[res]
            dict_feat['type'] = [self.residue_encoding[self.iScoreGraph.pdb.get('resName',chainID=res[0],resSeq=res[1])[0]]]

            attr = []
            for name in self.node_feature_str:
                attr += dict_feat[name]
            self.node.append(attr)

            self.node_info.append(self.iScoreGraph.pdb.get('chainID,resName',chainID=res[0],resSeq=res[1])[0] + [str(iRes)])

        self.num_nodes = len(self.node)
        self.num_node_features = len(self.node[0])

    def _get_edges(self):

        self.edge_index = self.iScoreGraph.edges
        self.edge_attr = []
        dict_attr = {}
        for iedge,_ in enumerate(self.edge_index):

            dict_attr['dist'] = self.edge_distance[iedge]
            dict_attr['polarity'] = self.edge_polarity[iedge]
            self.edge_attr.append([dict_attr[name] for name in self.edge_feature_str])

        self.num_edges = len(self.edge_index)
        self.num_edge_features = len(self.edge_attr[0])


    def _get_internal_edges(self):

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
                    edge_weightA.append(np.min(dist))

        edge_indexB,edge_weightB = [], []
        for i1 in range(nB-1):
            xyz1 = np.array(self.iScoreGraph.pdb.get('x,y,z',chainID=nodeB[i1][0],resSeq=nodeB[i1][1]))
            for i2 in range(i1+1,nB):
                xyz2 = np.array(self.iScoreGraph.pdb.get('x,y,z',chainID=nodeB[i2][0],resSeq=nodeB[i2][1]))
                dist = -2*np.dot(xyz1,xyz2.T) + np.sum(xyz1**2,axis=1)[:,None] + np.sum(xyz2**2,axis=1)
                if np.any(dist<self.internal_contact_distance):
                    edge_indexB.append([i1+nA,i2+nA])
                    edge_weightB.append(np.min(dist))


        self.internal_edge_index = np.vstack((edge_indexA,edge_indexB))
        self.internal_edge_weight = np.concatenate((edge_weightA,edge_weightB))





if __name__ == "__main__":

    pdb = 'data/pdb/1ATN.pdb'
    pdb = 'data/pdb/1ATN_149w.pdb'
    pssm = {'A':'./data/pssm/1ATN.A.pdb.pssm','B':'./data/pssm/1ATN.B.pdb.pssm'}
    graph = ResidueGraph(pdb,pssm)
    lgraph,g,lg = Graph.get_line_graph(graph)
