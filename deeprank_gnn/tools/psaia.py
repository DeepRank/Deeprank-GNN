# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 12:27:54 2012


Taken from PAIRPred http://combi.cs.colostate.edu/supplements/pairpred/

@author: root
"""
"""
pdb_id: 
analysis type: unbound
radii_file: /home/fayyaz/PSAIA-1.0/amac_data/chothia.radii
ASA_zSice: 0.25
ASA_rSolvent: 1.4
rasa_file: /home/fayyaz/PSAIA-1.0/amac_data/natural_asa.asa
CX_sRadius: 10
CX_vAtom: 20.1
radii_file: /home/fayyaz/PSAIA-1.0/amac_data/hydrophobicity.hpb
total ASA: 0
back-bone ASA: 0
side-chain ASA: 0
polar ASA: 0
non-polar ASA: 0

  chain id|   ch total ASA|  ch b-bone ASA| ch s-chain ASA|   ch polar ASA|
   0             1               2               3               4 
  ch n-polar ASA|    res id|  res name|      total ASA|     b-bone ASA|   
  5                   6       7               8               9   
  s-chain ASA|      polar ASA|    n-polar ASA|     total RASA|    b-bone RASA|
  10              11              12              13              14          
   s-chain RASA|     polar RASA|   n-polar RASA|    average DPX|      s_avg DPX| 
   15                  16          17              18                  19      
   s-ch avg DPX| s-ch s_avg DPX|        max DPX|        min DPX|     average CX| 
     20          21                  22                  23          24  
   s_avg CX|    s-ch avg CX|  s-ch s_avg CX|  max CX|    min CX| Hydrophobicity|                                        
   25              26          27               28           29          30
   
"""
import tempfile
import os
import glob
import pdb

PSAIA_PATH='/home/nico/programs/PSAIA-1.0'

if PSAIA_PATH not in  os.environ["PATH"]:
    print('Adding '+PSAIA_PATH+' to system path')
    os.environ["PATH"]+=os.pathsep+PSAIA_PATH

def getFileParts(fname):
    "Returns the parts of a file"
    (path, name) = os.path.split(fname)
    n=os.path.splitext(name)[0]
    ext=os.path.splitext(name)[1]
    return (path,n,ext)
        
def createPSAIAConfig(opath=''):
    """
    Creates a dictionary object for the configuration file for PSAIA
    opath is the path to the output directory
    """
    psaia_dict={}
    psaia_dict['analyze_bound']='1'
    psaia_dict['analyze_unbound']='1'
    psaia_dict['calc_asa']='1'    
    psaia_dict['z_slice']='0.25'
    psaia_dict['r_solvent']='1.4'
    psaia_dict['write_asa']='1' 
    psaia_dict['calc_rasa']='1'    
    psaia_dict['standard_asa']=PSAIA_PATH+'/amac_data/natural_asa.asa'
    psaia_dict['calc_dpx']='1'
    psaia_dict['calc_cx']='1'
    psaia_dict['cx_threshold']='10'
    psaia_dict['cx_volume']='20.1'
    psaia_dict['calc_hydro']='1'
    psaia_dict['hydro_file']=PSAIA_PATH+'/amac_data/hydrophobicity.hpb'
    psaia_dict['radii_filename']=PSAIA_PATH+'/amac_data/chothia.radii'
    psaia_dict['write_xml']='0'
    psaia_dict['write_table']='1'
    psaia_dict['output_dir']=opath
    return psaia_dict
    
"""    
analyze_bound:	1
analyze_unbound:	1
calc_asa:	1
z_slice:	0.25
r_solvent:	1.4
write_asa:	1
calc_rasa:	1
standard_asa:	/s/chopin/c/proj/protfun/arch/x86_64/psaia/PSAIA-1.0/amac_data/natural_asa.asa
calc_dpx:	1
calc_cx:	1
cx_threshold:	10
cx_volume:	20.1
calc_hydro:	1
hydro_file:	/s/chopin/c/proj/protfun/arch/x86_64/psaia/PSAIA-1.0/amac_data/hydrophobicity.hpb
radii_filename:	/s/chopin/c/proj/protfun/arch/x86_64/psaia/PSAIA-1.0/amac_data/chothia.radii
write_xml:	1
write_table:	1
output_dir:	/s/chopin/c/proj/protfun/arch/x86_64/psaia/PSAIA-1.0
"""

def writePSAIAConfgFile(cfname,d):
    """
    Writes the configuration dictionary created in createPSAIAConfig to file cfname
    """
    f=open(cfname,'w')
    for k in d:
        s=k+':\t'+d[k]+'\n'
        f.write(s)
    f.close()
    

def runPSAIA(fname,ofname=None):
    """
    Given a pdb file (fname), this function returns all the psaia features in 
    a dictionary. It removes all the temporary files created when the ofname is None.
    If the output filename is specified the files are retained
    """
    rmv=False
    if ofname is None:
        ofname=fname
        rmv=True
    out_file = tempfile.NamedTemporaryFile(suffix='.psaia');  out_file.close()        
    flist=out_file.name
    out_file = tempfile.NamedTemporaryFile(suffix='.psaia');  out_file.close() 
    fconfig=out_file.name
    #print flist,fconfig
    #Make list file
    fh=open(flist,'w')
    fh.write(fname)
    fh.close()
    #Make config file
    (cdir,xf,_)=getFileParts(fname)
    if cdir=='':
        cdir=os.getcwd()
    writePSAIAConfgFile(fconfig,createPSAIAConfig(cdir))
    cmdstr='yes y | psa '+fconfig+' '+flist    
    #pdb.set_trace()
    ecode=os.system(cmdstr)    
    os.remove(flist)
    os.remove(fconfig)
    xfname=glob.glob(os.path.join(cdir,xf+'*bound.tbl'))#[0]
    print(xfname)
    pdict=make_psaia_dict(xfname)
    
    if rmv: #cleanup
        for x in glob.glob(os.path.join(cdir,xf+'*bound.tbl')):
            os.remove(x)
    print('PSAIA Successful : ', cmdstr,ecode,xfname)
    return pdict
    #else:
    #    print('PSAIA Processing Failed.')
    #    return 0
    
    
def make_psaia_dict(filename):

    """
    Return a stride dictionary that maps (chainid, resname, resid) to
    aa, ss and accessibility, from a stride output file.
    @param filename: the stride output file
    @type filename: string
    """

    psaia = {}
    stnxt=0
    ln=0
    try:        
        for l in open(filename, "r"):
            ln=ln+1
            ls=l.split()
            if stnxt:     
                cid=ls[0]
                if cid=='*': #psaia replaces cid ' ' in pdb files with *
                    cid=' '
                resid=(cid,ls[6])#cid,resid,resname is ignored
                casa=map(float,ls[1:6])
                rasa=map(float,ls[8:13])
                rrasa=map(float,ls[13:18])
                rdpx=map(float,ls[18:24])
                rcx=map(float,ls[24:30])
                rhph=float(ls[-1])
                psaia[resid]=(casa,rasa,rrasa,rdpx,rcx,rhph)
                #pdb.set_trace()
            elif len(ls) and ls[0]=='chain': # the line containing 'chain' is the last line before real data starts
                stnxt=1
            
    except Exception as e:
        print('Error Processing psaia file: ' ,filename)
        print('Error occured while processing line:',ln)
        print(e)
        raise(e)
    return psaia

if __name__=="__main__":
    psaia=runPSAIA('./1AK4.pdb')
    #psaia=make_psaia_dict()