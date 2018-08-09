import os
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.HSExposure import HSExposureCA

import warnings
from Bio import BiopythonWarning

with warnings.catch_warnings():
    warnings.simplefilter('ignore',BiopythonWarning)
    from Bio import SearchIO

def get_bio_model(sqldb):
    sqldb.exportpdb('_tmp.pdb')
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('_tmp','_tmp.pdb')
    os.remove('_tmp.pdb')
    return structure[0]

def get_depth_res(model):

    rd = ResidueDepth(model)
    data = {}
    for k in list(rd.keys()):
        new_key = (k[0],k[1][1])
        data[new_key] = rd[k][0]
    return data

def get_hse(model):
    hse = HSExposureCA(model)
    data = {}
    for k in list(hse.keys()):
        new_key = (k[0],k[1][1])
        data[new_key] = hse[k][0]
    return data
