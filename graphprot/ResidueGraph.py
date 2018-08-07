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
        node_feature_str = 'chain, type, bsa, polarity, charge, pssm[20]',
        edge_feature_str = 'dist, polarity',
        internal_edge_feature_str = 'dist'):
        super().__init__()

        self.type = 'residue'
        self.pdb = pdb
        self.name = os.path.splitext(os.path.basename(pdb))[0]
        self.pssm = pssm
        self.contact_distance = contact_distance
        self.internal_contact_distance = internal_contact_distance

        feat,size = self.process_feat_str(node_feature_str)
        self.node_feature_str = feat
        self.node_feature_size = size

        feat,size = self.process_feat_str(edge_feature_str)
        self.edge_feature_str = feat
        self.edge_feature_size = size


        feat,size = self.process_feat_str(internal_edge_feature_str)
        self.internal_edge_feature_str = feat
        self.internal_edge_feature_size = size

        self.residue_charge = {'CYS':-0.64, 'HIS':-0.29, 'ASN':-1.22, 'GLN':-1.22, 'SER':-0.80, 'THR':-0.80, 'TYR':-0.80,
                                 'TRP':-0.79, 'ALA':-0.37, 'PHE':-0.37, 'GLY':-0.37,'ILE':-0.37,'VAL':-0.37,'MET':-0.37,
                                 'PRO':0.0,'LEU':-0.37, 'GLU':-1.37,'ASP':-1.37,'LYS':-0.36,'ARG':-1.65}


        self.residue_encoding = {'CYS':0, 'HIS':1, 'ASN':2, 'GLN':3, 'SER':4, 'THR':5, 'TYR':6, 'TRP':7,
                                 'ALA':8, 'PHE':9, 'GLY':10,'ILE':11,'VAL':12,'MET':13,'PRO':14,'LEU':15,
                                 'GLU':16,'ASP':17,'LYS':18,'ARG':20}


        self.residue_types = {'CYS':'polar','HIS':'polar','ASN':'polar','GLN':'polar','SER':'polar','THR':'polar','TYR':'polar','TRP':'polar',
                         'ALA':'apolar','PHE':'apolar','GLY':'apolar','ILE':'apolar','VAL':'apolar','MET':'apolar','PRO':'apolar','LEU':'apolar',
                         'GLU':'charged','ASP':'charged','LYS':'charged','ARG':'charged'}


        # get the residue graph
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
        self.get_nodes()
        self.get_edges()
        self.get_internal_edges()

    @staticmethod
    def process_feat_str(feat_str):
        feat,size = [],[]
        for name in feat_str.split(','):
            if '[' in name and ']' in name:
                s = name.split('[')
                feat.append(s[0].strip())
                s = s[1].split(']')
                size.append(int(s[0]))
            else:
                feat.append(name.strip())
                size.append(1)
        return feat,size


    def get_nodes(self):

        self.node = []
        self.node_info = []
        self.node_pos = []
        dict_feat = {}

        for iRes,res in enumerate(self.iScoreGraph.nodes):

            resName = self.iScoreGraph.pdb.get('resName',chainID=res[0],resSeq=res[1])[0]
            xyz = np.mean(self.iScoreGraph.pdb.get('x,y,z',chainID=res[0],resSeq=res[1]),0)

            dict_feat['chain'] = res[0]
            dict_feat['pssm'] = [float(p) for p in self.iScoreGraph.aligned_pssm[res]]
            dict_feat['bsa'] = self.BSA.bsa_data[res]
            dict_feat['type'] = [self.residue_encoding[resName]]
            dict_feat['charge'] = [self.residue_charge[resName]]
            dict_feat['polarity'] = [self.residue_types[resName]]

            attr = []
            for name in self.node_feature_str:
                if name in dict_feat:
                    attr += dict_feat[name]
                else:
                    print('Could not find node feature ', name)
            self.node.append(attr)
            self.node_info.append(self.iScoreGraph.pdb.get('chainID,resName',chainID=res[0],resSeq=res[1])[0] + [str(iRes)])
            self.node_pos.append(xyz)

        self.num_nodes = len(self.node)
        self.num_node_features = len(self.node[0])

    def get_edges(self):

        self.edge_index = self.iScoreGraph.edges
        self.edge_attr = []
        dict_attr = {}
        for iedge,_ in enumerate(self.edge_index):

            dict_attr['dist'] = self.edge_distance[iedge]
            dict_attr['polarity'] = self.edge_polarity[iedge]
            self.edge_attr.append([dict_attr[name] for name in self.edge_feature_str])

        self.num_edges = len(self.edge_index)
        self.num_edge_features = len(self.edge_attr[0])


    def get_internal_edges(self):

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
                    edge_weightA.append(np.mean(dist))

        edge_indexB,edge_weightB = [], []
        for i1 in range(nB-1):
            xyz1 = np.array(self.iScoreGraph.pdb.get('x,y,z',chainID=nodeB[i1][0],resSeq=nodeB[i1][1]))
            for i2 in range(i1+1,nB):
                xyz2 = np.array(self.iScoreGraph.pdb.get('x,y,z',chainID=nodeB[i2][0],resSeq=nodeB[i2][1]))
                dist = -2*np.dot(xyz1,xyz2.T) + np.sum(xyz1**2,axis=1)[:,None] + np.sum(xyz2**2,axis=1)
                if np.any(dist<self.internal_contact_distance):
                    edge_indexB.append([i1+nA,i2+nA])
                    edge_weightB.append(np.mean(dist))


        self.internal_edge_index = np.vstack((edge_indexA,edge_indexB))
        self.internal_edge_attr = np.concatenate((edge_weightA,edge_weightB))

    def _get_edge_polarity(self):

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

            ti = self.residue_types[resi]
            tj = self.residue_types[resj]

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


def plotly_graph_scatter3d(graph,out):

    import plotly.plotly as py
    import plotly.graph_objs as go


    edge_trace_list = []
    node_connect = [0]*len(graph.node_info)
    for edge_index, edge_attr in zip(graph.edge_index,graph.edge_attr):

        trace = go.Scatter3d(x=[],y=[],z=[],text=[],mode='lines', hoverinfo=None, line=go.Line(color='rgb(210,210,210)', width = 3+(1./edge_attr[0])))
        x0, y0, z0 = graph.node_pos[edge_index[0]]
        x1, y1, z1 = graph.node_pos[edge_index[1]]
        trace['x'] += [x0, x1, None]
        trace['y'] += [y0, y1, None]
        trace['z'] += [z0, z1, None]
        edge_trace_list.append(trace)
        node_connect[edge_index[0]] += 1
        node_connect[edge_index[1]] += 1

    internal_edge_trace_list = []
    for edge_index, edge_attr in zip(graph.internal_edge_index,graph.internal_edge_attr):

        trace = go.Scatter3d(x=[],y=[],z=[],text=[],mode='lines', hoverinfo=None, line=go.Line(color='rgb(110,110,110)', width = 3+1./edge_attr))

        x0, y0, z0 = graph.node_pos[edge_index[0]]
        x1, y1, z1 = graph.node_pos[edge_index[1]]
        trace['x'] += [x0, x1, None]
        trace['y'] += [y0, y1, None]
        trace['z'] += [z0, z1, None]
        internal_edge_trace_list.append(trace)


    node_trace = go.Scatter3d(x=[],y=[],z=[],text=[],mode='markers',hoverinfo='text',
                              marker=dict( colorscale='Electric', color=[], size=[], symbol='circle',
                                           line=dict(color='rgb(50,50,50)',width=2)))

    for pos,info,ncon in zip(graph.node_pos,graph.node_info,node_connect):
        node_trace['x'].append(pos[0])
        node_trace['y'].append(pos[1])
        node_trace['z'].append(pos[2])
        node_trace['text'].append(' '.join(info))
        node_trace['marker']['color'].append({'A':'rgb(227,28,28)','B':'rgb(0,102,255)'}[info[0]])
        node_trace['marker']['size'].append(5 + 15*np.tanh(ncon/5))

    fig = go.Figure(data=[*internal_edge_trace_list, *edge_trace_list, node_trace],
                 layout=go.Layout(
                    title='<br>Connection graph for %s' %graph.pdb,
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    py.iplot(fig, filename=out)

if __name__ == "__main__":

    pdb = 'data/pdb/1ATN/1ATN.pdb'
    pdb = 'data/pdb/1ATN/1ATN_143w.pdb'
    pssm = {'A':'./data/pssm/1ATN/1ATN.A.pdb.pssm','B':'./data/pssm/1ATN/1ATN.B.pdb.pssm'}
    graph = ResidueGraph(pdb,pssm)
    plotly_graph_scatter3d(graph,'graph_1ATN')
