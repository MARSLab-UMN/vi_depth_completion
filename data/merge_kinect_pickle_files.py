import os
import pickle
import sys

### FORMAT: python merge_kinect_pickle_files.py <file1> <file2> <file3> <file4> ... <fileN> output.pickle

if __name__ == '__main__':
    # assumes that each picle file contains a dictionary of {'train':->list of lists, 'test'->list os lists}
    num_sublists = -1

    assert len(sys.argv) > 3

    for index in range(1, len(sys.argv) - 1):
        with open(sys.argv[index], 'rb') as fp:
            data = pickle.load(fp)

        if num_sublists == -1:
            num_sublists = len(data['test'])
            final_map = {'test':[[] for i in range(num_sublists)], 'train':[[] for i in range(num_sublists)]}

        for i in range(num_sublists):
            final_map['test'][i].extend(data['test'][i])
            final_map['train'][i].extend(data['test'][i])


    with open(sys.argv[-1], 'wb') as f:
        pickle.dump(final_map, f)
