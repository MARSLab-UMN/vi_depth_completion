#!/bin/sh

### Evaluation script. If running for ScanNet or Kinect Azure, please first generate the pickle files
### by running the corresponding script in the data directory.
### A description of each command line option is provided the in main.py file.

python main.py --checkpoint ./checkpoints/depth_completion.ckpt \
  --surface_normal_checkpoint ./checkpoints/surface_normal_fpn_use_gravity.ckpt \
  --plane_detection_config_file ./plane_mask_detection/configs/R101_bs16_all_plane_normal.yaml \
  --train 0 --dataset_type demo --batch_size 1 --use_gravity 1 \
  --root ./demo_dataset --save ./demo_dataset


# w/o gravity, using DORN
# ScanNet
#python main.py --checkpoint ./checkpoints/depth_completion.ckpt \
#               --surface_normal_checkpoint ./checkpoints/surface_normal_dorn.ckpt \
#               --plane_detection_config_file ./plane_mask_detection/configs/R101_bs16_all_plane_normal.yaml \
#               --train 0 --dataset_type scannet --batch_size 16 --enriched_samples 100

# Azure Kinect
#python main.py --checkpoint ./checkpoints/depth_completion.ckpt \
#               --surface_normal_checkpoint ./checkpoints/surface_normal_dorn.ckpt \
#               --plane_detection_config_file ./plane_mask_detection/configs/R101_bs16_all_plane_normal.yaml \
#               --dataset_pickle_file ./data/all_kinect_triangulated_split.pkl \
#               --train 0 --dataset_type azure --batch_size 16 --enriched_samples 100 --skip_every_n_image_test 1

#
# w gravity, using FPN warping gravity alignment
# ScanNet
#python main.py --checkpoint ./checkpoints/depth_completion.ckpt \
#               --surface_normal_checkpoint ./checkpoints/surface_normal_fpn_use_gravity.ckpt \
#               --plane_detection_config_file ./plane_mask_detection/configs/R101_bs16_all_plane_normal.yaml \
#               --train 0 --dataset_type scannet --batch_size 16 --use_gravity 1 --enriched_samples 100

# Azure Kienct
#python main.py --checkpoint ./checkpoints/depth_completion.ckpt \
#               --surface_normal_checkpoint ./checkpoints/surface_normal_fpn_use_gravity.ckpt \
#               --plane_detection_config_file ./plane_mask_detection/configs/R101_bs16_all_plane_normal.yaml \
#               --dataset_pickle_file ./data/all_kinect_triangulated_split.pkl \
#               --train 0 --dataset_type azure --batch_size 16 --use_gravity 1 --enriched_samples 100 --skip_every_n_image_test 1
#
