import glob
import h5py
import os
import sys
import numpy as np


def add_target(graph_path, target_name, target_list, sep=' '):
    """Add a target to all the graphs contains in hdf5 files

    Args:
        graph_path (str, list(str)): either a directory containing all the hdf5 files,
                                     or a single hdf5 filename
                                     or a list of hdf5 filenames
        target_name (str): the name of the new target
        target_list (str): name of the file containing the data
        sep (str, optional): separator in target list. Defaults to ' '.

    Notes:
        The input target list should respect the following format :
        1ATN_xxx-1 0
        1ATN_xxx-2 1
        1ATN_xxx-3 0
        1ATN_xxx-4 0
    """

    target_dict = {}

    labels = np.loadtxt(target_list, delimiter=sep,
                        usecols=[0], dtype=str)
    values = np.loadtxt(target_list, delimiter=sep, usecols=[1])
    for label, value in zip(labels, values):
        target_dict[label] = value

    # if a directory is provided
    if os.path.isdir(graph_path):
        graphs = glob.glob('{}/*.hdf5'.format(graph_path))

    # if a single file is provided
    elif os.path.isfile(graph_path):
        graphs = [graph_path]

    # if a list of file is provided
    else:
        assert isinstance(graph_path, list)
        assert os.path.isfile(graph_path[0])

    for hdf5 in graphs:
        print(hdf5)
        try:
            f5 = h5py.File(hdf5, 'a')

            for model in f5.keys():
                try:
                    model_gp = f5['{}'.format(model)]

                    if 'score' not in model_gp:
                        model_gp.create_group('score')

                    group = f5['{}/score/'.format(model)]

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
