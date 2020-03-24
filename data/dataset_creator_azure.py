#!/usr/bin/env python3

### Creates a pickle file for a single Azure Kinect dataset.
### To create a pickle file for all Azure datasets, run this script on every dataset and
### merge the pickle files using merge_kinect_pickle_files.py script.

import argparse
import os
import pickle

parser = argparse.ArgumentParser(description='Dataset root directory')
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--check_for_percentage', action='store_true')
parser.add_argument('--percentage', type=float, default=8.0)
args = parser.parse_args()


def create_split_azure(split_name):
    root_dir = os.path.abspath(args.root)
    image_id = 0
    final_split = [[], [], [], [], []]
    while os.path.exists(os.path.join(root_dir, "color/color_%06d.png" % image_id)):
        # Make the train and test both contain all the images for now
        color_path = os.path.join(root_dir, "color/color_%06d.png" % image_id)
        depth_path = os.path.join(root_dir, "depth/depth_%06d.png" % image_id)
        gravity_path = os.path.join(root_dir, "gravity/gravity_%06d.txt" % image_id)
        mask_path = os.path.join(root_dir, "mask/mask_%06d.png" % image_id)
        normal_path = os.path.join(root_dir, "normal/normal_%06d.png" % image_id)

        ptcloud_file = os.path.join(root_dir, 'klt_tracks_triangulation/frame-{0:06d}-observed-pointcloud.txt'.format(image_id))
        if os.path.isfile(ptcloud_file):
            final_split[0].append(color_path)
            final_split[1].append(depth_path)
            final_split[2].append(gravity_path)
            final_split[3].append(mask_path)
            final_split[4].append(normal_path)
        image_id += 1

    print('Number of frames in the %s split: %d' % (split_name, len(final_split[0])))
    return final_split


def main():
    # Right now, the train is empty.
    final_dict = {'train': [[], [], [], [], []], 'test': create_split_azure('test')}
    dataset_name = os.path.basename(os.path.normpath(args.root))
    file_name = dataset_name + '_train_test_split_trinagulated.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(final_dict, f)
        print('Pickle file saved: ' + file_name)


if __name__ == "__main__":
    main()
