#!/usr/bin/env python3

import os
import pickle

ROOT_DIR = '/mars/mnt/dgx/FrameNet' #'/mnt/dgx/nyud_v2' #'/mnt/dgx/FrameNet'

def create_split(split_name):
    if split_name == 'train':
        start_scene_id = 0
        end_scene_id = 10
    elif split_name == 'test':
        start_scene_id = 0
        end_scene_id = 10

    image_id = 1
    frame_gap = 20
    scans_dir = 'scannet-small-frames' # "nyud-frames" # 'evaluation_scans' # 'scans' # 'scannet-frames'
    final_split = [[], [], []]
    num_frames = 0
    for scene_id in range(start_scene_id, end_scene_id):
        scan_id = 0
        while os.path.exists(os.path.join(ROOT_DIR, "%s/scene%04d_%02d" % (scans_dir, scene_id, scan_id))):
            frame_folder = os.path.join(ROOT_DIR, "%s/scene%04d_%02d" % (scans_dir, scene_id, scan_id))
            print('Working on folder: ', frame_folder)
            if split_name == 'train':
                frame_id = 0
            else:
                frame_id = 10
            while os.path.isfile(os.path.join(frame_folder, 'frame-%06d-normal.png' % frame_id)):
                color_path = os.path.join(frame_folder, 'frame-%06d-color.png' % frame_id)
                orient_path = os.path.join(frame_folder, 'frame-%06d-normal.png' % frame_id)
                mask_path = os.path.join(frame_folder, 'frame-%06d-orient-mask.png' % frame_id)
                final_split[0].append(color_path)
                final_split[1].append(orient_path)
                final_split[2].append(mask_path)
                frame_id += frame_gap
                num_frames += 1
            scan_id += 1
    print('Number of frames in the %s split: %d' % (split_name, num_frames))
    return final_split

def main():
    final_dict = {'train': create_split('train'), 'test': create_split('test')}
    with open('first10scenes_train_test_split.pkl', 'wb') as f:
        pickle.dump(final_dict, f)

if __name__ == "__main__":
    main()
