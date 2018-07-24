import os
import numpy as np
from collections import OrderedDict, namedtuple
from deeprank.features import AtomicFeature
from deeprank.tools import StructureSimilarity

from atomic_data import ELEMENTS
import pkg_resources

from Graph import Graph

import time

class AtomicGraph(Graph):

    def __init__(self,pdb=None,contact_distance=8.5,internal_contact_distance = 8.5,
                node_feature_str = 'chainID,name,x,y,z,eps,sig,charge',
                edge_feature_str = 'dist,coulomb,vanderwaals'):

        super().__init__()

        self.pdb = pdb
        self.type='atomic'
        self.name = os.path.splitext(os.path.basename(pdb))[0]
        self.FF = pkg_resources.resource_filename('deeprank.features','') + '/forcefield/'
        self.contact_distance = contact_distance
        self.internal_contact_distance = internal_contact_distance
        self.node_feature_str = [name.strip() for name in node_feature_str.split(',')]
        self.edge_feature_str = [name.strip() for name in edge_feature_str.split(',')]


        #t0 = time.time()
        self.atfeat = AtomicFeature(self.pdb,
                       contact_distance = self.contact_distance,
                       param_charge = self.FF + 'protein-allhdg5-4_new.top',
                       param_vdw    = self.FF + 'protein-allhdg5-4_new.param',
                       patch_file   = self.FF + 'patch.top')
        #print(' __ Init Graph %f' %(time.time()-t0))

        #t0 = time.time()
        self.atfeat.assign_parameters()
        #print(' __ Parameters %f' %(time.time()-t0))

        # only compute the pair interactions here
        #t0 = time.time()
        self.atfeat.evaluate_pair_interaction()
        #print(' __ Interaction %f' %(time.time()-t0))

        # get the pairs
        #t0 = time.time()
        self._get_pair()
        #print(' __ Get pair %f' %(time.time()-t0))

        # get edge attrs
        #t0 = time.time()
        self._get_edge_attribute()
        #print(' __ Get Edge %f' %(time.time()-t0))

        # get the edge within one chain
        #t0 = time.time()
        self._get_internal_edges()
        #print(' __ Get Internal Edge %f' %(time.time()-t0))

    def _get_pair(self):

        self.index_pairs_sql = self.atfeat.sqldb.get_contact_atoms(return_contact_pairs=True)
        self.AtomClass = namedtuple('Atom',self.node_feature_str,verbose=False)
        self.contact_atoms = OrderedDict()

        self.atoms,self.node,self.edge_index, self.pos = [],[],[],[]

        iNode = 0
        for indexA in self.index_pairs_sql.keys():
            if indexA not in self.contact_atoms:
                self.contact_atoms[indexA] = iNode
                feat = self.atfeat.sqldb.get(','.join(self.node_feature_str),rowID=indexA)[0]
                atom = self.AtomClass._make(feat)
                self.atoms.append(atom)
                self.node.append(self._hot_encoding(feat))
                self.pos.append([atom.x,atom.y,atom.z])
                iNode += 1

        for indexA,indicesB in self.index_pairs_sql.items():
            for indexB in indicesB:
                if indexB not in self.contact_atoms:
                    self.contact_atoms[indexB] = iNode
                    feat = self.atfeat.sqldb.get(','.join(self.node_feature_str),rowID=indexB)[0]
                    atom = self.AtomClass._make(feat)
                    self.atoms.append(atom)
                    self.node.append(self._hot_encoding(feat))
                    self.pos.append([atom.x,atom.y,atom.z])
                    iNode += 1

                self.edge_index.append([self.contact_atoms[indexA],self.contact_atoms[indexB]])

        self.num_nodes = len(self.node)
        self.num_node_features = len(self.node[0])
        self.num_edges = len(self.edge_index)
        self.pos = np.array(self.pos)

    @staticmethod
    def _hot_encoding(feat):
        feat[0] = {'A':0,'B':1}[feat[0]]
        atom_data = ELEMENTS[feat[1][0]]
        feat[1] = atom_data.number
        return feat

    def _get_edge_attribute(self):

        eps0 = 1
        c = 332.0636

        self.edge_attr = []
        for i,j in self.edge_index:

            dict_attr = {}

            # distance
            r = 0
            for x in ['x','y','z']:
                r += ( getattr(self.atoms[i],x)-getattr(self.atoms[j],x) )**2
            r = np.sqrt(r)
            dict_attr['dist'] = r

            # coulomb
            q1q2 = self.atoms[i].charge*self.atoms[j].charge
            coulomb = q1q2 * c / (eps0*r) * (1 - (r/self.contact_distance)**2 ) **2
            dict_attr['coulomb'] = coulomb

            # vdw terms
            sigma_avg = 0.5*(self.atoms[i].sig + self.atoms[j].sig)
            eps_avg = np.sqrt(self.atoms[i].eps*self.atoms[i].eps)

            # normal LJ potential
            evdw = 4.0 *eps_avg * (  (sigma_avg/r)**12  - (sigma_avg/r)**6 ) * self._prefactor_vdw(r)
            dict_attr['vanderwaals'] = evdw

            self.edge_attr.append([dict_attr[name] for name in self.edge_feature_str])

        self.num_edge_features = len(self.edge_attr[0])

    @staticmethod
    def _prefactor_vdw(r):
        """prefactor for vdw interactions."""

        r_off,r_on = 8.5,6.5
        if r > r_off:
            return 0.
        elif r < r_on:
            return 1.
        else:
            r2 = r**2
            return (r_off**2-r2)**2 * (r_off**2 - r2 - 3*(r_on**2 - r2)) / (r_off**2-r_on**2)**3


    def _get_internal_edges(self):

        tmp_node = np.array(self.node)
        feat = self.node_feature_str

        index_chain = feat.index('chainID')
        indexA = np.argwhere(tmp_node[:,index_chain] == 0).ravel()
        indexB = np.argwhere(tmp_node[:,index_chain] == 1).ravel()

        nodeA = tmp_node[indexA,:]
        nodeB = tmp_node[indexB,:]
        nA, nB = len(nodeA), len(nodeB)

        feat = self.node_feature_str
        index_pos = [feat.index(x) for x in ['x','y','z']]

        xyzA = nodeA[:,index_pos]
        edge_indexA, weightA = self._get_internal_edge_index(xyzA,self.internal_contact_distance)

        xyzB = nodeB[:,index_pos]
        edge_indexB, weightB = self._get_internal_edge_index(xyzB,self.internal_contact_distance)
        edge_indexB += nA

        self.internal_edge_index = np.vstack((edge_indexA,edge_indexB))
        self.internal_edge_weight = np.concatenate((weightA,weightB))

    @staticmethod
    def _get_internal_edge_index(xyz,cutoff):

        dist = -2*np.dot(xyz,xyz.T) + np.sum(xyz**2,axis=1)[:,None] + np.sum(xyz**2,axis=1)
        dist += np.eye(dist.shape[0])*2*cutoff
        index_edge = np.argwhere(dist<cutoff)

        n = 6
        index_weight = 1. / (dist[dist<cutoff]) ** 2

        return index_edge, index_weight

if __name__ == '__main__':

    from sklearn import manifold, datasets
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    import community

    pdb = './data/ref/1ATN.pdb'
    graph = AtomicGraph(pdb=pdb)
    nx = graph.toNX(internal_edge=False)
    #lgraph = Graph.get_line_graph(graph)

    for i,j in graph.internal_edge_index:
        if graph.node[i][0] != graph.node[j][0]:
            print('Issue with connection',i,j,graph.node[i][0],graph.node[j][0])

    part = community.best_partition(nx)
    size = len(set(part.values()))
    color = [v for k,v in part.items()]

    n_components = 2
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)

    Y = tsne.fit_transform(np.hstack((graph.pos,np.array(graph.node)[:,0].reshape(-1,1))))
    plt.scatter(Y[:,0],Y[:,1],c=color,cmap=plt.cm.tab10)

    for i in range(len(Y)):
        plt.text(Y[i,0],Y[i,1],str(color[i]))

    for i,j in graph.internal_edge_index:
        plt.plot([Y[i,0],Y[j,0]],[Y[i,1],Y[j,1]],c='black')
    plt.show()