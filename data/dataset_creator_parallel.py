#!/usr/bin/env python3

import os
import pickle
import numpy as np
from PIL import Image
from multiprocessing import Pool

ROOT_DIR = '/mars/mnt/dgx/FrameNet' #'/mnt/dgx/nyud_v2' #'/mnt/dgx/FrameNet'
PERCENTAGE = 8.0


def inc_depth_has_enough(incomp_depth_filename):
    I = np.asarray(Image.open(incomp_depth_filename))
    ratio = 100 * np.sum(I > 0) / I.size
    if ratio < PERCENTAGE:
        return False
    else:
        return True


def process_single_scene(input_ids):
    split_name, scene_id, scan_id = input_ids
    scans_dir = 'scannet-small-frames' # "nyud-frames" # 'evaluation_scans' # 'scans' # 'scannet-frames'
    final_split = [[], [], []]
    frame_gap = 10

    frame_folder = os.path.join(ROOT_DIR, "%s/scene%04d_%02d" % (scans_dir, scene_id, scan_id))
    print('Working on folder: ', frame_folder)
    if split_name == 'train':
        frame_id = 0
    else:
        frame_id = 10
    while os.path.isfile(os.path.join(frame_folder, 'frame-%06d-normal.png' % frame_id)):
        # Only add the frame if the incomplete depth exists

        if scene_id < 707:
            # Train scan
            di_path = frame_folder.replace('FrameNet/scannet-small-frames', 'ScanNet/scans')
        else:
            di_path = frame_folder.replace('FrameNet/scannet-small-frames', 'ScanNet/scans_test/scans')

        di_filename = os.path.join(di_path, 'incomplete_depth_abs_bugfix', 'depth-{0:06d}-incomplete.png'.format(frame_id))

        if os.path.isfile(di_filename) and inc_depth_has_enough(di_filename):
            color_path = os.path.join(frame_folder, 'frame-%06d-color.png' % frame_id)
            orient_path = os.path.join(frame_folder, 'frame-%06d-normal.png' % frame_id)
            mask_path = os.path.join(frame_folder, 'frame-%06d-orient-mask.png' % frame_id)
            final_split[0].append(color_path)
            final_split[1].append(orient_path)
            final_split[2].append(mask_path)

        frame_id += frame_gap

    return final_split


def create_split(split_name):
    num_parallel_workers = 40
    if split_name == 'train':
        start_scene_id = 0
        end_scene_id = 707
    elif split_name == 'test':
        start_scene_id = 707
        end_scene_id = 807

    scans_dir = 'scannet-small-frames' # "nyud-frames" # 'evaluation_scans' # 'scans' # 'scannet-frames'

    all_pairs = []
    for scene_id in range(start_scene_id, end_scene_id):
        scan_id = 0
        while os.path.exists(os.path.join(ROOT_DIR, "%s/scene%04d_%02d" % (scans_dir, scene_id, scan_id))):
            all_pairs.append((split_name, scene_id, scan_id))
            scan_id += 1

    pool = Pool(num_parallel_workers)
    all_outputs = pool.map(process_single_scene, all_pairs)

    final_split = [[], [], []]
    for l1, l2, l3 in all_outputs:
        final_split[0].extend(l1)
        final_split[1].extend(l2)
        final_split[2].extend(l3)

    print('Number of frames in the %s split: %d' % (split_name, len(final_split[0])))
    return final_split


def main():
    final_dict = {'train': create_split('train'), 'test': create_split('test')}
    with open('scannet_inc_depth_train_test_split.pkl', 'wb') as f:
        pickle.dump(final_dict, f)

if __name__ == "__main__":
    main()
