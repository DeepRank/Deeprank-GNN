#!/usr/bin/env python

import os
from h5xplorer.h5xplorer import h5xplorer
import h5x_menu

#baseimport = '/home/nico/Documents/projects/deeprank/DeepRank-GNN/deeprank_gnn/h5x/baseimport.py'
baseimport = os.path.dirname(
    os.path.abspath(__file__)) + "/baseimport.py"
app = h5xplorer(h5x_menu.context_menu,
                baseimport=baseimport, extended_selection=False)
