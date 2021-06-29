import glob
import h5py
import sys
import numpy as np 

def add_target(graph_path, target_name, target_list, sep=' '):

    target_dict = {}

    # Converts the target list into a dictionary
    # The input target list should respect the following format :
    # 1ATN_xxx-1 0
    # 1ATN_xxx-2 1
    # 1ATN_xxx-3 0
    # 1ATN_xxx-4 0

    labels = np.loadtxt(target_list, delimiter=sep, usecols=[0], dtype=np.str)
    values = np.loadtxt(target_list, delimiter=sep, usecols=[1])
    for label, value in zip(labels, values):
        target_dict[label] = value

    first_model = True

    for hdf5 in glob.glob('{}/*.hdf5'.format(graph_path)):
        try:
            f5 = h5py.File(hdf5, 'a')

            for model in f5.keys():
                try :
                    model_gp = f5['{}'.format(model)]

                    if 'score' not in model_gp :
                        model_gp.create_group('score')

                    group=f5['{}/score/'.format(model)]

                    if first_model is True :
                        print('The target {} does not exist. Create it'.format(target_name))
                        first_model = False

                    if target_name in group.keys():
                        # Delete the target if it already existed                                                                                                                    
                        del group[target_name]

                    # Create the target                                                                                                                                              
                    group[target_name] = target_dict[model]

                except:
                    print('no graph for {}'.format(model))
                    
            f5.close()

        except:
            print('no graph for {}'.format(hdf5))   
            
