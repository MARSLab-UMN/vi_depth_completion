import os
import pickle
from PIL import Image
import numpy as np


ROOT_DIR = '/mars/mnt/dgx/FrameNet'
NORMALS_DIR = '/mars/mnt/dgx/FrameNet/DORN_acos_bs16_inference/'  #'/mars/mnt/dgx/FrameNet/L2_inference/'
PLANES_DIR = '/mars/mnt/dgx/FrameNet/plane_inference/'
OUTPUT_FILE_NAME = 'scannet_depth_completion_split_acos.pkl'

INCOMPLETE_DEPTH_DIRECTORY='incomplete_depth_dorn_acos'

CHECK_FOR_MIN_PERCENT = True
MIN_PERCENT = 8


def check_min_perc(scannet_scene_base, scene_name, frame_id):
    inc_depth_filename = os.path.join(scannet_scene_base, scene_name, '{0}/depth-{1:06d}-incomplete.png'.format(INCOMPLETE_DEPTH_DIRECTORY, frame_id))
    depths = np.asarray(Image.open(inc_depth_filename))
    valid_depths = (depths > 0).astype(np.float)
    num_valid_depths = np.sum(valid_depths)
    return ((100.0 * num_valid_depths / depths.size) >= MIN_PERCENT)


def create_split(split_name):
    if split_name == 'train':
        start_scene_id = 0
        end_scene_id = 707
    elif split_name == 'test':
        start_scene_id = 707
        end_scene_id = 807

    image_id = 1
    frame_gap = 10
    scans_dir = 'scannet-small-frames' # "nyud-frames" # 'evaluation_scans' # 'scans' # 'scannet-frames'
    final_split = [[], [], []]
    num_frames = 0
    for scene_id in range(start_scene_id, end_scene_id):
        scan_id = 0
        while os.path.exists(os.path.join(ROOT_DIR, "%s/scene%04d_%02d" % (scans_dir, scene_id, scan_id))):
            frame_folder = os.path.join(ROOT_DIR, "%s/scene%04d_%02d" % (scans_dir, scene_id, scan_id))
            #print('Working on folder: ', frame_folder)
            if split_name == 'train':
                frame_id = 0
            else:
                frame_id = 10
            while os.path.isfile(os.path.join(frame_folder, 'frame-%06d-normal.png' % frame_id)):
                color_path = os.path.join(frame_folder, 'frame-%06d-color.png' % frame_id)
                orient_path = os.path.join(frame_folder, 'frame-%06d-normal.png' % frame_id)
                mask_path = os.path.join(frame_folder, 'frame-%06d-orient-mask.png' % frame_id)
                plane_path = os.path.join(PLANES_DIR, '{0}/scene{1:04d}_{2:02d}/frame-{3:06d}-instance_segmentation.png'.format(PLANES_DIR, scene_id, scan_id, frame_id))
                # Check if this frame has the incomplete depth. If so, then it also has the inference normal and other related.
                if scene_id < 707:
                    scannet_scene_base = '/mars/mnt/dgx/ScanNet/scans/'
                else:
                    scannet_scene_base = '/mars/mnt/dgx/ScanNet/scans_test/scans'
                scene_name = 'scene%04d_%02d' % (scene_id, scan_id)
                incomplete_depth_path = os.path.join(scannet_scene_base, scene_name, '{0}/depth-{1:06d}-incomplete.png'.format(INCOMPLETE_DEPTH_DIRECTORY, frame_id))
                if os.path.isfile(color_path) and os.path.isfile(orient_path) and os.path.isfile(mask_path) and os.path.isfile(plane_path) and os.path.isfile(incomplete_depth_path):
                    should_add = False
                    if not CHECK_FOR_MIN_PERCENT:
                        should_add = True
                    elif check_min_perc(scannet_scene_base, scene_name, frame_id):
                        # In this case CHECK_FOR_MIN_PERCENT is True.
                        should_add = True

                    if should_add:
                        final_split[0].append(color_path)
                        final_split[1].append(orient_path)
                        final_split[2].append(mask_path)
                frame_id += frame_gap
                num_frames += 1
            scan_id += 1

        print('Scene {}: {}'.format(scene_id, len(final_split[0])))

    print('Number of frames in the %s split: %d' % (split_name, num_frames))
    return final_split

def main():
    final_dict = {'train': create_split('train'), 'test': create_split('test')}
    with open(OUTPUT_FILE_NAME, 'wb') as f:
        pickle.dump(final_dict, f)

if __name__ == "__main__":
    main()

