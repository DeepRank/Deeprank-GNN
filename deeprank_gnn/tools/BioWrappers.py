import os
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import ResidueDepth, get_surface, residue_depth
from Bio.PDB.HSExposure import HSExposureCA

import warnings
from Bio import BiopythonWarning

import tempfile

with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonWarning)
    from Bio import SearchIO

from time import time


def get_bio_model(pdbfile):
    """Get the model

    Args:
        pdbfile (str): pdbfile

    Returns:
        [type]: Bio object
    """
    parser = PDBParser()
    structure = parser.get_structure('_tmp', pdbfile)
    return structure[0]


def get_depth_res(model):
    """Get the residue Depth

    Args:
        model (bio model): model of the strucrture

    Returns:
        dict: depth res
    """
    rd = ResidueDepth(model)

    data = {}
    t0 = time()
    for k in list(rd.keys()):
        new_key = (k[0], k[1][1])
        data[new_key] = rd[k][0]

    return data


def get_depth_contact_res(model, contact_res):
    """Get the residue Depth

    Args:
        model (bio model): model of the strucrture
        contact_res (list): list of contact residues

    Returns:
        dict: depth res
    """

    surface = get_surface(model)
    data = {}
    for r in contact_res:
        chain = model[r[0]]
        res = chain[r[1]]
        data[r] = residue_depth(res, surface)
    return data


def get_hse(model):
    """Get the hydrogen surface exposure

    Args:
        model (bio model): model of the strucrture

    Returns:
        dict: hse data
    """

    hse = HSExposureCA(model)
    data = {}
    for k in list(hse.keys()):
        new_key = (k[0], k[1][1])

        x = hse[k]
        if x[2] is None:
            x = list(x)
            x[2] = 0.0
            x = tuple(x)

        data[new_key] = x
    return data
