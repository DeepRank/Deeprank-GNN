import os
from deeprank.tools import StructureSimilarity
import numpy as np
import networkx as nx
from collections import OrderedDict
import time

class Graph(object):

    def __init__(self):

        self.type = None
        self.name = None

        self.num_nodes = None
        self.num_node_features = None
        self.node = None

        self.pos = None

        self.num_edges = None
        self.num_edge_features = None
        self.edge_attr = None
        self.edge_index = None

        self.irmsd = None
        self.lrmsd = None
        self.fnat = None
        self.dockQ = None
        self.binclass = None

        self.node_feature_str = None
        self.edge_feature_str = None

        self.internal_edge_index = None
        self.internal_edge_weight = None

        self.node_info = None

    @classmethod
    def get_line_graph(cls,g):

        # create a nx graph
        # that should be the default ....
        #t0 = time.time()
        nx_graph = nx.Graph()
        for index in g.edge_index:
            i,j = index
            nx_graph.add_edge(i,j)
        nx_line_graph = nx.line_graph(nx_graph)
        #print(' __ networkx %f' %(time.time()-t0))

        #make a new instance of the class
        # and copy some attributes
        lg = cls()
        for k in ['type','name','irmsd','lrmsd','fnat','dockQ','binclass']:
            lg.__dict__[k] = g.__dict__[k]

        # new number of nodes/edges
        lg.num_nodes = nx_line_graph.number_of_nodes()
        lg.num_edges = nx_line_graph.number_of_edges()

        # the node feature are now edge_attr + node_feat1 + node_feat2
        nodefeat1 = [x+'_1' for x in g.node_feature_str]
        nodefeat2 = [x+'_2' for x in g.node_feature_str]
        lg.node_feature_str = g.edge_feature_str + nodefeat1 + nodefeat2

        # crete a dict of node (i,e edge index) new node index
        dict_node = OrderedDict()
        lg.node, lg.pos = [], []
        #t0 = time.time()
        for iN,key in enumerate(nx_line_graph.nodes.keys()):

            # map edge_index <-> new node index
            dict_node[key] = iN

            # the attribute of the normal graph edge
            ind_attr = g.edge_index.index(list(key))
            edge_attr = g.edge_attr[ind_attr]

            # the feature of the connected node in the old graph
            node_feat_1 = g.node[key[0]]
            node_feat_2 = g.node[key[1]]

            # create the line graph node feature
            lg.node.append(edge_attr+node_feat_1+node_feat_2)
            pos = 0.5*(g.pos[key[0]]+g.pos[key[1]])
            lg.pos.append(pos)
        #print(' __ Nodes %f' %(time.time()-t0))

        # numbr of node feature
        lg.num_node_features = len(lg.node[0])

        # create the edge
        lg.edge_index = []
        lg.edge_feature_str = g.node_feature_str
        lg.edge_attr = []

        #t0 = time.time()
        for key in nx_line_graph.edges.keys():
            lg.edge_index.append([dict_node[key[0]],dict_node[key[1]]])

            index_node = list(set(key[0]).intersection(key[1]))[0]
            lg.edge_attr.append(g.node[index_node])
        #print(' __ Edges %f' %(time.time()-t0))

        return lg


    def toNX(self,internal_edge=False):

        nx_graph = nx.Graph()

        for iN,node in enumerate(self.node):
            nx_graph.add_node(iN,data=node)

        if not internal_edge:
            for index in self.edge_index:
                i,j = index
                nx_graph.add_edge(i,j)
        else:
            for index in self.internal_edge_index:
                i,j = index
                nx_graph.add_edge(i,j)

        return nx_graph

    def score(self,decoy,ref):

        ref_name = os.path.splitext(os.path.basename(ref))[0]
        sim = StructureSimilarity(decoy,ref)
        self.lrmsd = sim.compute_lrmsd_fast(method='svd',lzone=ref_name+'.lzone')
        self.irmsd = sim.compute_irmsd_fast(method='svd',izone=ref_name+'.izone')
        self.fnat = sim.compute_Fnat_fast(ref_pairs=ref_name+'.refpairs')
        self.dockQ = sim.compute_DockQScore(self.fnat,self.lrmsd,self.irmsd)
        self.binclass = not self.irmsd < 4.0

    def export_hdf5(self,f5):

        keys = ['type','name','num_nodes','num_edges','pos',
                'num_node_features','num_edge_features',
                'node','edge_index','edge_attr','internal_edge_index','internal_edge_attr',
                'irmsd','lrmsd','fnat','dockQ']

        grp = f5.create_group(self.name)
        grp.attrs['node_feature'] = np.void([bytes(name,encoding='utf-8') for name in self.node_feature_str])
        grp.attrs['edge_attr'] = np.void([bytes(name,encoding='utf-8') for name in self.edge_feature_str])

        for k in keys:
            if self.__dict__[k] is not None:
                grp.create_dataset(k,data=self.__dict__[k])




