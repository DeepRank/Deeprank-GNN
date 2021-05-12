# %matplotlib inline

import community
import networkx as nx
import torch

from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

from torch_geometric.data import Data
from deeprank_gnn.community_pooling import *
from deeprank_gnn.Graph import Graph


def tsne_graph(grp, method):

    import plotly.offline as py
    py.init_notebook_mode(connected=True)

    g = Graph()
    g.h52nx(None, None, molgrp=grp)
    g.plotly_2d(offline=True, iplot=False, method=method)


def graph3d(grp):

    import plotly.offline as py
    py.init_notebook_mode(connected=True)

    g = Graph()
    g.h52nx(None, None, molgrp=grp)
    g.plotly_3d(offline=True, iplot=False)
