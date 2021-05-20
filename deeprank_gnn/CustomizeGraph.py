import glob
import h5py
import sys

def add_target(self, graph_path, target_name, target_list, sep=' '):

    # Converts the target list into a dictionary
    # The input target list should respect the following format :
    # 1ATN_xxx-1.pdb 0
    # 1ATN_xxx-2.pdb 1
    # 1ATN_xxx-3.pdb 0
    # 1ATN_xxx-4.pdb 0
    labels, values = np.genfromtxt(target_list, delimiter=sep, unpack=True)
    for label, value in zip(labels, values):
        self.target_dict[label] = value
        
    first_model = True

    for hdf5 in glob.glob('{}/*.hdf5'.format(graph_path)):
        try:
            f5 = h5py.File(hdf5, 'a')
            for model in f5.keys():
                group=f5['{}/score/'.format(model)]
                        
                # Delete the target if it already existed
                try: 
                    del group[target_name]
                    
                except :
                    if first_model is True :
                        print('The target {} does not exist. Create it'.format(target_name))
                        first_model = False     

                # Create the target     
                group[target_name] = self.target_dict[model]

                f5.close()

        except:
            print('no graph for {}'.format(hdf5))

