import sys
import h5py
import pandas as pd
import numpy as np
 
def hdf5_to_csv(hdf5_path):
        
        hdf5 = h5py.File(hdf5_path,'r+')
        name = hdf5_path.split('.')[0]
        
        first = True
        for epoch in hdf5.keys():
                for dataset in hdf5['{}'.format(epoch)].keys():
                        mol = hdf5['{}/{}/mol'.format(epoch, dataset)]
                        epoch_lst = [epoch] * len(mol)
                        dataset_lst = [dataset] * len(mol)

                        outputs = hdf5['{}/{}/outputs'.format(epoch, dataset)]
                        targets = hdf5['{}/{}/targets'.format(epoch, dataset)]
                        if len(targets) == 0:
                                targets = 'n'*len(mol)

                        # This section is specific to the binary class
                        # it adds the raw output, i.e. probabilities to belong to the class 0 and the class 1, to the prediction hdf5
                        # This way, binary information can be transformed back to continuous data and used for ranking
                        if 'raw_output' in hdf5['{}/{}'.format(epoch, dataset)].keys():
                                if first :
                                        header = ['epoch', 'set', 'model', 'targets', 'prediction', 'raw_prediction_0', 'raw_prediction_1']
                                        output_file = open('{}.csv'.format(name), 'w')
                                        output_file.write(','+','.join(header)+'\n')
                                        output_file.close()
                                        first = False
                                # probability of getting 0         
                                outputs_0 = hdf5['{}/{}/raw_output'.format(epoch, dataset)][()][:,0]
                                # probability of getting 1
                                outputs_1 = hdf5['{}/{}/raw_output'.format(epoch, dataset)][()][:,1]
                                dataset_df = pd.DataFrame(list(zip(epoch_lst, dataset_lst, mol, targets, outputs, outputs_0, outputs_1)), columns = header)
                        
                        else:
                                if first :
                                        header = ['epoch', 'set', 'model', 'targets', 'prediction']
                                        output_file = open('{}.csv'.format(name), 'w')
                                        output_file.write(','+','.join(header)+'\n')
                                        output_file.close()
                                        first = False
                                dataset_df = pd.DataFrame(list(zip(epoch_lst, dataset_lst, mol, targets, outputs)), columns = header)
                        
                        dataset_df.to_csv('{}.csv'.format(name), mode='a', header=False)
        

if __name__ == "__main__":	
        
	if len(sys.argv) != 2 :
        	print ("""\n
This scripts converts the hdf5 output files of GraphProt into csv files
                
Usage: 
python hdf5_to_csv.py file.hdf5
""")
                
	else: 
                try: 
                        hdf5_path = sys.argv[1]
                        hdf5_to_csv(hdf5_path)
                        
                except:
                        print('Please make sure that your input file if a HDF5 file')