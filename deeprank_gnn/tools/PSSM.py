import os
import numpy as np


def read_PSSM_data(fname):
    """Read the PSSM data."""

    f = open(fname,'r')
    data = f.readlines()
    f.close()

    filters = (lambda x: len(x.split())>0, lambda x: x.split()[0].isdigit())
    return list(map(lambda x: x.split(),list(filter(lambda x: all(f(x) for f in filters), data))))


def PSSM_aligned(pssmfiles,style='HADDOCK'):

    resmap = {
    'A' : 'ALA', 'R' : 'ARG', 'N' : 'ASN', 'D' : 'ASP', 'C' : 'CYS', 'E' : 'GLU', 'Q' : 'GLN',
    'G' : 'GLY', 'H' : 'HIS', 'I' : 'ILE', 'L' : 'LEU', 'K' : 'LYS', 'M' : 'MET', 'F' : 'PHE',
    'P' : 'PRO', 'S' : 'SER', 'T' : 'THR', 'W' : 'TRP', 'Y' : 'TYR', 'V' : 'VAL',
    'B' : 'ASX', 'U' : 'SEC', 'Z' : 'GLX'
    }

    pssm, ic = {}, {}
    for chain in ['A','B']:

        data = read_PSSM_data(pssmfiles[chain])
        for l in data :
            if style == 'res':
                resi = int(l[0])
                resn = resmap[l[1]]
            elif style == 'seq':
                resi = int(l[2])
                resn = resmap[l[3]]
            pssm[(chain,resi,resn)] = list(map(lambda x: float(x), l[4:24]))
            ic[(chain,resi,resn)] = float(l[24])

    return pssm, ic

def get_pssm_data(node,pssm):
    return pssm[node] if node in pssm else [0]*20

def get_ic_data(node,ic):
    return ic[node] if node in ic else 0.
