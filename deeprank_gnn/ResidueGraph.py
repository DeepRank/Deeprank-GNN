import os
import numpy as np
import shutil
import torch 
from time import time
import networkx as nx

from pdb2sql import interface

from .tools import BioWrappers, PSSM, BSA
from .Graph import Graph


class ResidueGraph(Graph):

    def __init__(self, pdb=None, pssm=None,
                 contact_distance=8.5, internal_contact_distance=3,
                 pssm_align='res'):
        """Class from which Residue features are computed

        Args:
            pdb (str, optional): path to pdb file. Defaults to None.
            pssm (str, optional): path to pssm file. Defaults to None.
            contact_distance (float, optional): cutoff distance for external edges. Defaults to 8.5.
            internal_contact_distance (int, optional): cutoff distance for internal edges. Defaults to 3.
            pssm_align (str, optional): [description]. Defaults to 'res'.
        """
        super().__init__()

        self.type = 'residue'
        self.pdb = pdb
        self.name = os.path.splitext(os.path.basename(pdb))[0]

        if pssm is not None:
            self.pssm, self.ic = PSSM.PSSM_aligned(
                pssm, style=pssm_align)
        else:
            self.pssm, self.ic = None, None

        self.contact_distance = contact_distance
        self.internal_contact_distance = internal_contact_distance

        self.residue_charge = {'CYS': -0.64, 'HIS': -0.29, 'ASN': -1.22, 'GLN': -1.22, 'SER': -0.80, 'THR': -0.80, 'TYR': -0.80,
                               'TRP': -0.79, 'ALA': -0.37, 'PHE': -0.37, 'GLY': -0.37, 'ILE': -0.37, 'VAL': -0.37, 'MET': -0.37,
                               'PRO': 0.0, 'LEU': -0.37, 'GLU': -1.37, 'ASP': -1.37, 'LYS': -0.36, 'ARG': -1.65}

        self.residue_names = {'CYS': 0, 'HIS': 1, 'ASN': 2, 'GLN': 3, 'SER': 4, 'THR': 5, 'TYR': 6, 'TRP': 7,
                              'ALA': 8, 'PHE': 9, 'GLY': 10, 'ILE': 11, 'VAL': 12, 'MET': 13, 'PRO': 14, 'LEU': 15,
                              'GLU': 16, 'ASP': 17, 'LYS': 18, 'ARG': 19}

        self.residue_polarity = {'CYS': 'polar', 'HIS': 'polar', 'ASN': 'polar', 'GLN': 'polar', 'SER': 'polar', 'THR': 'polar', 'TYR': 'polar', 'TRP': 'polar',
                                 'ALA': 'apolar', 'PHE': 'apolar', 'GLY': 'apolar', 'ILE': 'apolar', 'VAL': 'apolar', 'MET': 'apolar', 'PRO': 'apolar', 'LEU': 'apolar',
                                 'GLU': 'neg_charged', 'ASP': 'neg_charged', 'LYS': 'neg_charged', 'ARG': 'pos_charged'}

        self.pssm_pos = {'CYS': 4, 'HIS': 8, 'ASN': 2, 'GLN': 5, 'SER': 15, 'THR': 16, 'TYR': 18, 'TRP': 17,
                         'ALA': 0, 'PHE': 13, 'GLY': 7, 'ILE': 9, 'VAL': 19, 'MET': 12, 'PRO': 14, 'LEU': 10,
                         'GLU': 6, 'ASP': 3, 'LYS': 11, 'ARG': 1}

        self.polarity_encoding = {
            'apolar': 0, 'polar': 1, 'neg_charged': 2, 'pos_charged': 3}
        #self.edge_polarity_encoding, iencod = {}, 0
        ##for k1, v1 in self.polarity_encoding.items():
        ##for k2, v2 in self.polarity_encoding.items():
        ##key = tuple(np.sort([v1, v2]))
        ##if key not in self.edge_polarity_encoding:
        ##self.edge_polarity_encoding[key] = iencod
        #iencod += 1

        # check if external execs are installed
        self.check_execs()

        # create the sqldb
        db = interface(self.pdb)

        # get the graphs
        t0 = time()
        self.get_graph(db)
        print('Graph %f' % (time()-t0))

        # get the nodes/edge attributes
        t0 = time()
        self.get_node_features(db)
        print('Node %f' % (time()-t0))

        t0 = time()
        self.get_edge_features()
        print('Edge %f' % (time()-t0))

        # close the db
        db._close()

    def check_execs(self):
        """Checks if all the required external execs are installed.

        Raises:
            OSError: if execs not found
        """

        execs = {'msms': 'http://mgltools.scripps.edu/downloads#msms'}

        for e, inst in execs.items():
            if shutil.which(e) is None:
                raise OSError(e, ' is not installed see ',
                              inst, ' for details')

    def get_graph(self, db):
        """Gets the interface graph nodes and edges given the 
        internal and external distance threshold

        Args:
            db ([type]): sqldb database
        """
        self.nx = nx.Graph()
        self.nx.edge_index = []
        self.res_contact_pairs = db.get_contact_residues(
            cutoff=self.contact_distance, return_contact_pairs=True)

        # get all the nodes
        all_nodes = self._get_all_valid_nodes(
            self.res_contact_pairs, self.pssm)

        # create the interface edges
        for key, val in self.res_contact_pairs.items():
            if key in all_nodes:
                for v in val:
                    if v in all_nodes:
                        d = self._get_edge_distance(key, v, db)
                        self.nx.add_edge(key, v, dist=d, type=bytes(
                            'interface', encoding='utf-8'))
                    else:
                        print('WARNING: {} is not a valid node'.format(v))
            else:
                print('WARNING: {} is not a valid node'.format(key))

        # get the internal edges
        edges, dist = self.get_internal_edges(db)
        for e, d in zip(edges, dist):
            e0, e1 = tuple(e[0]), tuple(e[1])
            if e0 in all_nodes and e1 in all_nodes:
                self.nx.add_edge(e0, e1, dist=d, type=bytes(
                    'internal', encoding='utf-8'))
            else:
                print('WARNING: {} {} is not a valid edge'.format(e0, e1))

    @staticmethod
    def _get_all_valid_nodes(res_contact_pairs, pssm, verbose=False):
        """Filters out invalid nodes based on 2 criteria:
            - contains unrecognized residue(s)
            - contains residue(s) absent form the pssm matrix

        Args:
            res_contact_pairs ([type]): list of contact pairs 
            pssm ([type]): pssm file
            verbose (bool, optional): [description]. Defaults to False.

        Returns:
            Valid list of contact pairs
        """
        valid_res = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
            'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'ASX', 'SEC', 'GLX']

        # tag the non residues
        keys_to_pop = []
        for res in res_contact_pairs.keys():
            if res[2] not in valid_res:
                keys_to_pop.append(res)
                # res_contact_pairs.pop(res,None)
                Warning('--> Residue ', res, ' not valid')

        # tag the ones that are not in PSSM
        if pssm is not None:
            for res in list(res_contact_pairs.keys()):
                if res not in pssm:
                    keys_to_pop.append(res)
                    # res_contact_pairs.pop(res,None)
                    Warning('--> Residue ', res,
                            ' not found in PSSM file')

        # Remove the residue
        for res in keys_to_pop:
            if res in res_contact_pairs:
                res_contact_pairs.pop(res, None)

        # get a list of residues of chain B
        # automatically remove the ones that are not proper residues
        # and the ones that are not in the PSSM
        nodesB = []
        for k, reslist in list(res_contact_pairs.items()):

            for res in reslist:
                if res[2] in valid_res:
                    if pssm is None:
                        nodesB += [res]
                    elif res in pssm:
                        nodesB += [res]
                else:
                    if verbose:
                        print('removed node', res)
        nodesB = sorted(set(nodesB))

        # make a list of nodes
        return list(res_contact_pairs.keys()) + nodesB

    def get_node_features(self, db):
        """Assigns features to each node
        """

        # get the BSA of the residues
        t0 = time()
        bsa_calc = BSA.BSA(self.pdb, db)
        bsa_calc.get_structure()
        bsa_calc.get_contact_residue_sasa(
            cutoff=self.contact_distance)
        bsa_data = bsa_calc.bsa_data
        #print('_BSA %f' %(time()-t0))

        # biopython data
        t0 = time()
        model = BioWrappers.get_bio_model(db.pdbfile)
        #print('_Model %f' %(time()-t0))

        #t0 = time()
        ResDepth = BioWrappers.get_depth_contact_res(
            model, self.nx.nodes)
        #print('_RD %f' %(time()-t0))

        #t0 = time()
        HSE = BioWrappers.get_hse(model)
        #print('_HSE %f' %(time()-t0))

        # loop over all the nodes
        for node_key in self.nx.nodes:

            chainID = node_key[0]
            resName = node_key[2]
            resSeq = int(node_key[1])

            self.nx.nodes[node_key]['chain'] = {
                'A': 0, 'B': 1}[chainID]
            self.nx.nodes[node_key]['pos'] = np.mean(
                db.get('x,y,z', chainID=chainID, resSeq=resSeq), 0)
            self.nx.nodes[node_key]['type'] = self.onehot(
                self.residue_names[resName], len(self.residue_names))

            self.nx.nodes[node_key]['charge'] = self.residue_charge[resName]
            self.nx.nodes[node_key]['polarity'] = self.onehot(
                self.polarity_encoding[self.residue_polarity[resName]], len(self.polarity_encoding))

            self.nx.nodes[node_key]['bsa'] = bsa_data[node_key]

            if self.pssm is not None:
                data = PSSM.get_pssm_data(node_key, self.pssm)
                self.nx.nodes[node_key]['pssm'] = data
                self.nx.nodes[node_key]['cons'] = data[self.pssm_pos[resName]]
                self.nx.nodes[node_key]['ic'] = PSSM.get_ic_data(
                    node_key, self.ic)

            self.nx.nodes[node_key]['depth'] = ResDepth[node_key] if node_key in ResDepth else 0
            bio_key = (chainID, resSeq)
            self.nx.nodes[node_key]['hse'] = HSE[bio_key] if bio_key in HSE else (
                0, 0, 0)

    def get_edge_features(self):
        """Assigns distance feature to each edge
        """
        node_keys = list(self.nx.nodes)

        for e in self.nx.edges:
            node1, node2 = e
            #self.nx.edges[node1, node2]['polarity'] = self._get_edge_polarity(
            #    node1, node2)
            self.nx.edge_index.append(
                [node_keys.index(node1), node_keys.index(node2)])

    def get_internal_edges(self, db):
        """Gets internal edges and the associated distances 
        """
        nodesA, nodesB = [], []
        for n in self.nx.nodes:
            if n[0] == 'A':
                nodesA.append(n)
            elif n[0] == 'B':
                nodesB.append(n)

        edgesA, distA = self._get_internal_edges_chain(
            nodesA, db, self.internal_contact_distance)
        edgesB, distB = self._get_internal_edges_chain(
            nodesB, db, self.internal_contact_distance)

        return edgesA+edgesB, distA+distB

    def _get_internal_edges_chain(self, nodes, db, cutoff):
        """Gets internal edges and the associated distances (min) per chain

        Args:
            nodes ([type]): list of nodes
            db ([type]): sqldb database
            cutoff ([type]): internal edge cutoff

        Returns:
            edges between two nodes, 
            and the minimum distance between those two nodes
        """
        lnodes = list(nodes)
        nn = len(nodes)
        edges, dist = [], []
        for i1 in range(nn):
            xyz1 = np.array(
                db.get('x,y,z', chainID=nodes[i1][0], resSeq=nodes[i1][1]))
            for i2 in range(i1+1, nn):
                xyz2 = np.array(
                    db.get('x,y,z', chainID=nodes[i2][0], resSeq=nodes[i2][1]))
                d2 = -2*np.dot(xyz1, xyz2.T) + np.sum(xyz1**2,
                                                      axis=1)[:, None] + np.sum(xyz2**2, axis=1)
                if np.any(d2 < cutoff**2):
                    edges.append((nodes[i1], nodes[i2]))
                    dist.append(np.sqrt(np.min(d2)))

        return edges, dist

    def _get_internal_edges_chain_mean(self, nodes, db, cutoff):
        """Gets internal edges and the associated distance (mean) per chain

        Args:
            nodes ([type]): list of nodes
            db ([type]): sqldb database
            cutoff ([type]): internal edge cutoff

        Returns:
            edges between two nodes, 
            and the mean distance between those two nodes
        """
        lnodes = list(nodes)
        nn = len(nodes)
        edges, dist, centers = [], [], []

        for i1 in range(nn):
            centers.append(
                np.mean(db.get('x,y,z', chainID=nodes[i1][0], resSeq=nodes[i1][1]), 0))

        centers = np.array(centers)
        distances = -2*np.dot(centers, centers.T) + np.sum(
            centers**2, axis=1)[:, None] + np.sum(centers**2, axis=1)

        indexes = np.argwhere(distances < 2*cutoff)

        for i, j in indexes:
            if i != j:
                edges.append((nodes[i], nodes[j]))
                dist.append(distances[i, j])

        return edges, dist

    def _get_edge_polarity(self, node1, node2):
        """Gets edge polarity

        Args:
            node1 
            node2 
        """
        v1 = self.nx.nodes[node1]['polarity']
        v2 = self.nx.nodes[node2]['polarity']

        key = tuple(np.sort([v1, v2]))
        return self.edge_polarity_encoding[key]

    def _get_edge_distance(self, node1, node2, db):

        # pos1 = self.nx.nodes[node1]['pos']
        # pos2 = self.nx.nodes[node2]['pos']
        # return np.linalg.norm(pos1-pos2)
        xyz1 = np.array(
            db.get('x,y,z', chainID=node1[0], resSeq=node1[1]))
        xyz2 = np.array(
            db.get('x,y,z', chainID=node2[0], resSeq=node2[1]))
        d2 = -2*np.dot(xyz1, xyz2.T) + np.sum(xyz1**2,
                                              axis=1)[:, None] + np.sum(xyz2**2, axis=1)
        return np.sqrt(np.min(d2))


    def onehot(self, idx, size):
        """One hot encoder
        """
        onehot = torch.zeros(size)
        # Fill the one-hot encoded sequence with 1 at the corresponding idx
        onehot[idx] = 1
        return np.array(onehot)
