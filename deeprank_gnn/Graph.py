import os
from pdb2sql import StructureSimilarity
import numpy as np
import networkx as nx
from collections import OrderedDict
import time
from deeprank_gnn.tools.embedding import manifold_embedding
import community
import markov_clustering as mc
import h5py


class Graph(object):

    def __init__(self):
        """Class that performs graph level action

            - get score for the graph (lrmsd, irmsd, fnat, capri_class, bin_class and dockQ)
            - networkx object (graph) to hdf5 format
            - networkx object (graph) from hdf5 format
        """
        self.name = None
        self.nx = None
        self.score = {'irmsd': None, 'lrmsd': None, 'capri_class': None,
                      'fnat': None, 'dockQ': None, 'bin_class': None}

    def get_score(self, ref):
        """Assigns scores (lrmsd, irmsd, fnat, dockQ, bin_class, capri_class) to a protein graph

        Args:
            ref (path): path to the reference structure required to compute the different score
        """

        ref_name = os.path.splitext(os.path.basename(ref))[0]
        sim = StructureSimilarity(self.pdb, ref)
        
        # Input pre-computed zone files
        if os.path.exists(ref_name+'.lzone'):
            self.score['lrmsd'] = sim.compute_lrmsd_fast(
            method='svd', lzone=ref_name+'.lzone')
            self.score['irmsd'] = sim.compute_irmsd_fast(
            method='svd', izone=ref_name+'.izone')
        
        # Compute zone files
        else: 
            self.score['lrmsd'] = sim.compute_lrmsd_fast(
            method='svd')
            self.score['irmsd'] = sim.compute_irmsd_fast(
            method='svd')

        self.score['fnat'] = sim.compute_fnat_fast()
        self.score['dockQ'] = sim.compute_DockQScore(
            self.score['fnat'], self.score['lrmsd'], self.score['irmsd'])
        self.score['bin_class'] = self.score['irmsd'] < 4.0
        
        self.score['capri_class'] = 5
        for thr, val in zip([6.0, 4.0, 2.0, 1.0], [4, 3, 2, 1]):
            if self.score['irmsd'] < thr:
                self.score['capri_class'] = val
        
    def nx2h5(self, f5):
        """Converts Networkx object to hdf5 format

        Args:
            f5 ([type]): hdf5 file
        """
        # group for the graph
        grp = f5.create_group(self.name)

        # store the nodes
        data = np.array(list(self.nx.nodes)).astype('S')
        grp.create_dataset('nodes', data=data)

        # store node features
        node_feat_grp = grp.create_group('node_data')
        feature_names = list(list(self.nx.nodes.data())[0][1].keys())
        for feat in feature_names:
            data = [v for _, v in nx.get_node_attributes(
                self.nx, feat).items()]
            node_feat_grp.create_dataset(feat, data=data)

        edges, internal_edges = [], []
        edge_index, internal_edge_index = [], []

        edge_feat = list(list(self.nx.edges.data())[0][2].keys())
        edge_data, internal_edge_data = {}, {}
        for feat in edge_feat:
            edge_data[feat] = []
            internal_edge_data[feat] = []

        node_key = list(self.nx.nodes)

        for e in self.nx.edges:

            edge_type = self.nx.edges[e]['type'].decode('utf-8')
            ind1, ind2 = node_key.index(e[0]), node_key.index(e[1])

            if edge_type == 'interface':
                edges.append(e)
                edge_index.append([ind1, ind2])

                for feat in edge_feat:
                    edge_data[feat].append(self.nx.edges[e][feat])

            elif edge_type == 'internal':
                internal_edges.append(e)
                internal_edge_index.append([ind1, ind2])

                for feat in edge_feat:
                    internal_edge_data[feat].append(
                        self.nx.edges[e][feat])

        # store the edges
        data = np.array(edges).astype('S')
        grp.create_dataset('edges', data=data)

        data = np.array(internal_edges).astype('S')
        grp.create_dataset('internal_edges', data=data)

        # store the edge index
        grp.create_dataset('edge_index', data=edge_index)
        grp.create_dataset('internal_edge_index',
                           data=internal_edge_index)

        # store the edge attributes
        edge_feat_grp = grp.create_group('edge_data')
        internal_edge_feat_grp = grp.create_group(
            'internal_edge_data')

        for feat in edge_feat:
            edge_feat_grp.create_dataset(feat, data=edge_data[feat])
            internal_edge_feat_grp.create_dataset(
                feat, data=internal_edge_data[feat])

        # store the score
        score_grp = grp.create_group('score')
        for k, v in self.score.items():
            if v is not None:
                score_grp.create_dataset(k, data=v)

    def h52nx(self, f5name, mol, molgrp=None):
        """Converts Hdf5 file to Networkx object

        Args:
            f5name (str): hdf5 file
            mol (str): molecule name
            molgrp ([type], optional): hdf5[molecule]. Defaults to None.
        """
        if molgrp is None:

            f5 = h5py.File(f5name, 'r')
            molgrp = f5[mol]
            self.name = mol
            self.pdb = mol+'.pdb'

        else:
            self.name = molgrp.name
            self.pdb = self.name+'.pdb'

        # creates the graph
        self.nx = nx.Graph()

        # get nodes
        nodes = molgrp['nodes'][()].astype('U').tolist()
        nodes = [tuple(n) for n in nodes]

        # get node features
        node_keys = list(molgrp['node_data'].keys())
        node_feat = {}
        for key in node_keys:
            node_feat[key] = molgrp['node_data/'+key][()]

        # add nodes
        for iN, n in enumerate(nodes):
            self.nx.add_node(n)
            for k, v in node_feat.items():
                v = v.reshape(-1, 1) if v.ndim == 1 else v
                v = v[iN, :]
                v = v[0] if len(v) == 1 else v
                self.nx.nodes[n][k] = v

        # get edges
        edges = molgrp['edges'][()].astype('U').tolist()
        edges = [(tuple(e[0]), tuple(e[1])) for e in edges]

        # get edge data
        edge_key = list(molgrp['edge_data'].keys())
        edge_feat = {}
        for key in edge_key:
            edge_feat[key] = molgrp['edge_data/'+key][()]

        # add edges
        for iedge, e in enumerate(edges):
            self.nx.add_edge(e[0], e[1])
            for k, v in edge_feat.items():
                v = v.reshape(-1, 1) if v.ndim == 1 else v
                v = v[iedge, :]
                v = v[0] if len(v) == 1 else v
                self.nx.edges[e[0], e[1]][k] = v

        # get internal edges
        edges = molgrp['internal_edges'][()].astype(
            'U').tolist()
        edges = [(tuple(e[0]), tuple(e[1])) for e in edges]

        # get edge data
        edge_key = list(molgrp['internal_edge_data'].keys())
        edge_feat = {}
        for key in edge_key:
            edge_feat[key] = molgrp['internal_edge_data/' +
                                    key][()]

        # add edges
        for iedge, e in enumerate(edges):
            self.nx.add_edge(e[0], e[1])
            for k, v in edge_feat.items():
                v = v.reshape(-1, 1) if v.ndim == 1 else v
                v = v[iedge, :]
                v = v[0] if len(v) == 1 else v
                self.nx.edges[e[0], e[1]][k] = v

        # add score
        self.score = {}
        score_key = list(molgrp['score'].keys())
        for key in score_key:
            self.score[key] = molgrp['score/'+key][()]

        # add cluster
        if 'clustering' in molgrp:
            self.clusters = {}
            for method in list(molgrp['clustering'].keys()):
                self.clusters[method] = molgrp['clustering/' +
                                               method+'/depth_0'][()]

        if molgrp is None:
            f5.close()

    def plotly_2d(self, out=None, offline=False, iplot=True, method='louvain'):
        """Plots the interface graph in 2D

        Args:
            out ([type], optional): output name. Defaults to None.
            offline (bool, optional): Defaults to False.
            iplot (bool, optional): Defaults to True.
            method (str, optional): 'mcl' of 'louvain'. Defaults to 'louvain'.
        """
        if offline:
            import plotly.offline as py
        else:
            import chart_studio.plotly as py

        import plotly.graph_objs as go
        import matplotlib.pyplot as plt

        pos = np.array(
            [v.tolist() for _, v in nx.get_node_attributes(self.nx, 'pos').items()])
        pos2D = manifold_embedding(pos)
        dict_pos = {n: p for n, p in zip(self.nx.nodes, pos2D)}
        nx.set_node_attributes(self.nx, dict_pos, 'pos2D')

        # remove interface edges for clustering
        gtmp = self.nx.copy()
        ebunch = []
        for e in self.nx.edges:
            typ = self.nx.edges[e]['type']
            if isinstance(typ, bytes):
                typ = typ.decode('utf-8')
            if typ == 'interface':
                ebunch.append(e)
        gtmp.remove_edges_from(ebunch)

        try:

            raw_cluster = self.clusters[method]
            cluster = {}
            node_key = list(self.nx.nodes.keys())

            for inode, index_cluster in enumerate(raw_cluster):
                cluster[node_key[inode]] = index_cluster
        except:

            print(
                'No cluster in the group. Computing cluster now with method : %s' % method)

            if method == 'louvain':
                cluster = community.best_partition(gtmp)

            elif method == 'mcl':
                matrix = nx.to_scipy_sparse_matrix(gtmp)
                # run MCL with default parameters
                result = mc.run_mcl(matrix)
                mcl_clust = mc.get_clusters(result)    # get clusters
                cluster = {}
                node_key = list(self.nx.nodes.keys())
                for ic, c in enumerate(mcl_clust):
                    for node in c:
                        cluster[node_key[node]] = ic

        # get the colormap for the clsuter line
        ncluster = np.max([v for _, v in cluster.items()])+1
        cmap = plt.cm.nipy_spectral
        N = cmap.N
        cmap = [cmap(i) for i in range(N)]
        cmap = cmap[::int(N/ncluster)]
        cmap = 'plasma'

        edge_trace_list, internal_edge_trace_list = [], []
        node_connect = {}

        for edge in self.nx.edges:

            edge_type = self.nx.edges[edge[0],
                                      edge[1]]['type'].decode('utf-8')
            if edge_type == 'internal':
                trace = go.Scatter(x=[], y=[], text=[], mode='lines', hoverinfo=None,  showlegend=False,
                                   line=go.scatter.Line(color='rgb(110,110,110)', width=3))
            elif edge_type == 'interface':
                trace = go.Scatter(x=[], y=[], text=[], mode='lines', hoverinfo=None,  showlegend=False,
                                   line=go.scatter.Line(color='rgb(210,210,210)', width=1))

            x0, y0 = self.nx.nodes[edge[0]]['pos2D']
            x1, y1 = self.nx.nodes[edge[1]]['pos2D']

            trace['x'] += (x0, x1, None)
            trace['y'] += (y0, y1, None)

            if edge_type == 'internal':
                internal_edge_trace_list.append(trace)
            elif edge_type == 'interface':
                edge_trace_list.append(trace)

            for i in [0, 1]:
                if edge[i] not in node_connect:
                    node_connect[edge[i]] = 1
                else:
                    node_connect[edge[i]] += 1
        node_trace_A = go.Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text',
                                  marker=dict(color='rgb(227,28,28)', size=[],
                                              line=dict(color=[], width=4, colorscale=cmap)))
        # 'rgb(227,28,28)'
        node_trace_B = go.Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text',
                                  marker=dict(color='rgb(0,102,255)', size=[],
                                              line=dict(color=[], width=4, colorscale=cmap)))
        # 'rgb(0,102,255)'
        node_trace = [node_trace_A, node_trace_B]

        for node in self.nx.nodes:

            index = self.nx.nodes[node]['chain']
            pos = self.nx.nodes[node]['pos2D']

            node_trace[index]['x'] += (pos[0],)
            node_trace[index]['y'] += (pos[1],)
            node_trace[index]['text'] += (
                '[Clst:' + str(cluster[node]) + '] ' + ' '.join(node),)

            nc = node_connect[node]
            node_trace[index]['marker']['size'] += (
                5 + 15*np.tanh(nc/5),)
            node_trace[index]['marker']['line']['color'] += (
                cluster[node],)

        fig = go.Figure(data=[*internal_edge_trace_list, *edge_trace_list, *node_trace],
                        layout=go.Layout(
            title='<br>tSNE connection graph for %s' % self.pdb,
            titlefont=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002)],
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        if iplot:
            py.iplot(fig, filename=out)
        else:
            py.plot(fig)

    def plotly_3d(self, out=None, offline=False, iplot=True):
        """Plots interface graph in 3D

        Args:
            out ([type], optional): [description]. Defaults to None.
            offline (bool, optional): [description]. Defaults to False.
            iplot (bool, optional): [description]. Defaults to True.
        """
        if offline:
            import plotly.offline as py
        else:
            import chart_studio.plotly as py

        import plotly.graph_objs as go

        edge_trace_list, internal_edge_trace_list = [], []
        node_connect = {}

        for edge in self.nx.edges:

            edge_type = self.nx.edges[edge[0],
                                      edge[1]]['type'].decode('utf-8')
            if edge_type == 'internal':
                trace = go.Scatter3d(x=[], y=[], z=[], text=[], mode='lines', hoverinfo=None,  showlegend=False,
                                     line=go.scatter3d.Line(color='rgb(110,110,110)', width=5))
            elif edge_type == 'interface':
                trace = go.Scatter3d(x=[], y=[], z=[], text=[], mode='lines', hoverinfo=None,  showlegend=False,
                                     line=go.scatter3d.Line(color='rgb(210,210,210)', width=2))

            x0, y0, z0 = self.nx.nodes[edge[0]]['pos']
            x1, y1, z1 = self.nx.nodes[edge[1]]['pos']

            trace['x'] += (x0, x1, None)
            trace['y'] += (y0, y1, None)
            trace['z'] += (z0, z1, None)

            if edge_type == 'internal':
                internal_edge_trace_list.append(trace)
            elif edge_type == 'interface':
                edge_trace_list.append(trace)

            for i in [0, 1]:
                if edge[i] not in node_connect:
                    node_connect[edge[i]] = 1
                else:
                    node_connect[edge[i]] += 1
        node_trace_A = go.Scatter3d(x=[], y=[], z=[], text=[], mode='markers', hoverinfo='text',
                                    marker=dict(color='rgb(227,28,28)', size=[], symbol='circle',
                                                line=dict(color='rgb(50,50,50)', width=2)))

        node_trace_B = go.Scatter3d(x=[], y=[], z=[], text=[], mode='markers', hoverinfo='text',
                                    marker=dict(color='rgb(0,102,255)', size=[], symbol='circle',
                                                line=dict(color='rgb(50,50,50)', width=2)))

        node_trace = [node_trace_A, node_trace_B]

        for node in self.nx.nodes:

            index = self.nx.nodes[node]['chain']
            pos = self.nx.nodes[node]['pos']

            node_trace[index]['x'] += (pos[0],)
            node_trace[index]['y'] += (pos[1],)
            node_trace[index]['z'] += (pos[2], )
            node_trace[index]['text'] += (' '.join(node),)

            nc = node_connect[node]
            node_trace[index]['marker']['size'] += (
                5 + 15*np.tanh(nc/5), )

        fig = go.Figure(data=[*node_trace, *internal_edge_trace_list, *edge_trace_list],
                        layout=go.Layout(
                        title='<br>Connection graph for %s' % self.pdb,
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        if iplot:
            py.iplot(fig, filename=out)
        else:
            py.plot(fig)


if __name__ == "__main__":
    import h5py
    graph = Graph()
    graph.h52nx('1AK4_residue.hdf5', '1ATN')
    graph.plotly_2d('1ATN')
