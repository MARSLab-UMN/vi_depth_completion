import argparse
import logging
import numpy as np
import os
import sys
import torch
import torch.autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

from dataset import *
import network_run
from networks.depth_completion import *
from networks.surface_normal import *
from networks.surface_normal_dorn import *

from plane_mask_detection.maskrcnn_benchmark.config import cfg
from plane_mask_detection.demo.predictor import COCODemo


MAX_DEPTH_DIFF_MULTIPLIER = 10
MAX_DEPTH = 10
# Angle in degrees
MEAN_NORMAL_ANGLE_DIFF_THR = 20


# all_normals Nx3 tensor
def mean_normal(all_normals: torch.Tensor):
    assert(len(all_normals.shape) == 2)
    assert(all_normals.shape[1] == 3)

    mean = torch.mean(all_normals, dim=0)
    return F.normalize(mean, dim=0)


# Determines the mean normal with outlier rejection
def mean_normal_ranasc(all_normals: torch.Tensor, angle_threshold_degrees: float=20.0, num_hypotheses: int=300):
    assert(len(all_normals.shape) == 2)
    assert(all_normals.shape[1] == 3)
    num_normals = all_normals.shape[0]
    # Assume the normals are of the shape of (N, 3)
    random_indices = np.random.permutation(np.r_[0:num_normals])[0:min(num_hypotheses, num_normals)]

    all_dots = torch.mm(all_normals[random_indices, :], torch.transpose(all_normals, 0, 1))
    all_dots = torch.clamp(all_dots, -1.0, 1.0)
    all_angles = torch.acos(all_dots) * (180.0 / np.pi)
    close_normals = all_angles < angle_threshold_degrees
    num_inliers = torch.sum(close_normals, dim=1)

    best_inlier_mask = close_normals[torch.argmax(num_inliers).item(), :]

    inlier_normals = all_normals[best_inlier_mask, :]
    mean_normal_value = mean_normal(inlier_normals)

    # Compute the angle differences.
    dots = torch.mm(inlier_normals, mean_normal_value[:, None])
    dots = torch.clamp(dots, -1, 1)
    angle_differences = torch.acos(dots) * (180 / np.pi)

    # Now get the normal for those that are close to this normal and compute the average.
    return mean_normal_value, angle_differences, best_inlier_mask


# Given the plane normal, estimate the plane offset using known points on the plane.
# normal: A vector of 3 elements corresponding to the plane normal.
# C_p_f: A 3xN array of point locations at the current camera frame.
def plane_offset_ransac(normal: torch.Tensor, C_p_f: torch.Tensor, distance_threshold: float=1.e-1, min_inliers: int=1, num_hypotheses: int=300):
    num_points = C_p_f.shape[1]

    if num_points == 0:
        return 0, 0

    effective_num_hypotheses = min(num_hypotheses, num_points)
    if num_points <= num_hypotheses:
        random_indices = np.r_[0:num_points]
    else:
        random_indices = np.random.permutation(np.r_[0:num_points])[0:effective_num_hypotheses]
    all_dots = torch.mm(normal[None, :], C_p_f).squeeze()
    if all_dots.nelement() == 1:
        if min_inliers == 1:
            return -all_dots, 1
        else:
            return 0, 0

    hypo_offset = -all_dots[random_indices]
    point_to_plane_distances = torch.abs(hypo_offset[..., None] + all_dots[None, ...])
    assert(point_to_plane_distances.shape[0] == hypo_offset.nelement())

    inliers = point_to_plane_distances < distance_threshold
    num_inliers = torch.sum(inliers, dim=1)
    assert(num_inliers.nelement() == effective_num_hypotheses)
    best_inlier_index = torch.argmax(num_inliers).item()

    if num_inliers[best_inlier_index] < min_inliers:
        return 0, 0

    best_inlier_mask = inliers[best_inlier_index, :]

    plane_offset = -torch.mean(all_dots[best_inlier_mask])
    return plane_offset, torch.sum(best_inlier_mask).item()


# Fills the depth image using the plane equation and plane mask.
# plane_eq: An array of four elements representing the plane equation.
# mask: The binary mask indicating the pixels that belong to this plane.
# homogeneous: The homogeneous location of each pixel.
# depth_image: The output (and already initialized) depth image. This is so that we can call this function multiple
# times for different planes but on the same depth image.
def generate_depth_from_plane(plane_eq: torch.Tensor, mask: torch.Tensor, homogeneous: torch.Tensor,
                              depth_image: torch.Tensor, mean_depth, best_inlier_mask: torch.Tensor):
    normal = plane_eq[0:3].flatten()
    #dots = torch.mm(homogeneous, normal[:, None])
    dots = torch.sum(homogeneous * normal[None, None, :], dim=2)

    mask2 = mask & (torch.abs(dots) > 1e-3)
    depth_values = -plane_eq[3] / dots
    actual_values = depth_values[mask2]

    if (torch.sum(actual_values > mean_depth * MAX_DEPTH_DIFF_MULTIPLIER) / actual_values.nelement() > 0.05 or
        torch.sum(actual_values > MAX_DEPTH) / actual_values.nelement() > 0):
        return False
    if torch.sum(actual_values<0) / actual_values.nelement() > 0.0:
        return False

    depth_image[mask2] = actual_values
    return True


def extract_plane_images_from_normal_image(normal_image: torch.Tensor, mask: torch.Tensor, depth: torch.Tensor,
                                           homogeneous: torch.Tensor):
    # First, collect all unique masks
    classes = torch.unique(mask)
    num_classes = torch.max(classes) + 1
    if num_classes == 1:
        # The mask could not detect anything, so just return the input.
        return normal_image, depth

    # Convert [3, H, W] to [H, W, 3].
    normal_image = normal_image.permute([1, 2, 0])
    planes_depth_image = depth.clone()

    class_with_no_depth = 0

    planes_normal_image = normal_image.clone()

    for cls in classes:
        if cls == 0:
            continue
        mask_for_class = mask == cls
        normals = normal_image[mask_for_class]

        surface_normal, angle_differences, best_inlier_mask = mean_normal_ranasc(normals)

        # Here we need to set the mask_for_class pixels which have value of 1 based on
        # the inlier mask. Hence the trick here:
        mask_for_class[mask_for_class] = best_inlier_mask
        # TODO: Can potentially try to find other planes within the "outliers".

        # If the mean normal is very different from the normal vectors inside the plane, then
        # it probably means that the plane mask is not correct.
        if torch.mean(torch.abs(angle_differences)) > MEAN_NORMAL_ANGLE_DIFF_THR:
            continue

        planes_normal_image = torch.where(mask_for_class[..., None],
                                          surface_normal[None, None, :],
                                          planes_normal_image)
        if cls == class_with_no_depth:
            continue
        # Now complete the depth given the plane normal and homogeneous location, as well as the depth values.
        valid_depths_mask = mask_for_class & (depth > 0)
        point_cloud = homogeneous[valid_depths_mask] * depth[valid_depths_mask][:, None]
        mean_depth = torch.mean(depth[valid_depths_mask])

        plane_offset, num_inliers = plane_offset_ransac(surface_normal, point_cloud.T)
        # If no inliers, we will not have depth for this plane.
        if num_inliers == 0:
            continue
        plane_params = torch.zeros(4).to(normal_image.device)
        plane_params[0:3] = surface_normal
        plane_params[3] = plane_offset
        plane_valid = generate_depth_from_plane(plane_params, mask_for_class, homogeneous, planes_depth_image, mean_depth, best_inlier_mask)


    # To retain the depth values, for the pixels that did not fall on the planes, move the values from depth to planes_depth_image
    valid_depths = depth > 0
    planes_depth_image[valid_depths] = depth[valid_depths]

    planes_normal_image = planes_normal_image.permute([2, 0, 1])
    return planes_normal_image, planes_depth_image


def ParseCmdLineArguments():
    parser = argparse.ArgumentParser(description='MARS CNN Script')
    parser.add_argument('--checkpoint', action='append',
                        help='Location of the checkpoints to evaluate.')
    parser.add_argument('--train', type=int, default=1,
                        help='If set to nonzero train the network, otherwise will evaluate.')
    parser.add_argument('--save', type=str, default='',
                        help='The path to save the network checkpoints and logs.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--root', type=str, default='/mars/mnt/dgx/FrameNet')
    parser.add_argument('--epoch', type=int, default=0,
                        help='The epoch to resume training from.')
    parser.add_argument('--iter', type=int, default=0,
                        help='The iteration to resume training from.')
    parser.add_argument('--dataset_pickle_file', type=str, default='./data/scannet_depth_completion_split.pkl')
    parser.add_argument('--dataloader_test_workers', type=int, default=16)
    parser.add_argument('--dataloader_train_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1.e-4)
    parser.add_argument('--save_every_n_iteration', type=int, default=1000,
                        help='Save a checkpoint every n iterations (iterations reset on new epoch).')
    parser.add_argument('--save_every_n_epoch', type=int, default=1,
                        help='Save a checkpoint on the first iteration of every n epochs (independent of iteration).')
    parser.add_argument('--enable_multi_gpu', type=int, default=0,
                        help='If nonzero, use all available GPUs.')
    parser.add_argument('--skip_every_n_image_test', type=int, default=40,
                        help='Skip every n image in the test split.')
    parser.add_argument('--eval_test_every_n_iterations', type=int, default=1000,
                        help='Evaluate the network on the test set every n iterations when in training.')
    parser.add_argument('--resnet_arch', type=int, default=101,
                        help='ResNet architecture for ModifiedFPN (101 or 50)')
    parser.add_argument('--surface_normal_checkpoint', type=str, default='',
                        help='Surface normal checkpoint path is a required field.')
    parser.add_argument('--plane_detection_config_file', type=str, default='',
                        help='Plane detection config path is a required field.')
    parser.add_argument('--enriched_samples', type=int, default=200,
                        help='Number of samples used to enrich the depth.')
    parser.add_argument('--dataset_type', type=str, default='scannet',
                        help='The dataset loader fromat. Closely related to the pickle file (scannet, nyu, azure).')
    parser.add_argument('--use_gravity', type=int, default=0,
                        help='Use with or without gravity.')
    return parser.parse_args()


class RunDepthCompletion(network_run.DefaultImageNetwork):
    def __init__(self, arguments, train_dataloader, test_dataloader, network_class_creator, use_gravity=False):
        super(RunDepthCompletion, self).__init__(arguments, train_dataloader, test_dataloader,
                                                 network_class=network_class_creator,
                                                 estimates_depth=True)
        self.use_gravity = use_gravity
        if self.use_gravity:
            self.surface_normal_cnn = SurfaceNormalPrediction(fc_img=np.array([202., 202.])).cuda()
        else:
            self.surface_normal_cnn = SurfaceNormalDORN().cuda()
        self.plane_masks_extraction = None

    def eval_mode(self):
        self.surface_normal_cnn.eval()
        self.cnn.eval()

    def load_plane_extraction_network_from_file(self, config_file):
        cfg.merge_from_file(config_file)
        self.plane_masks_extraction = COCODemo(cfg, min_image_size=240, confidence_threshold=0.9)

    def load_surface_normal_network_from_file(self, checkpoint):
        state = self.surface_normal_cnn.state_dict()
        state.update(torch.load(checkpoint))
        self.surface_normal_cnn.load_state_dict(state)

    def _call_cnn(self, input_batch):
        ds = input_batch['sparse_depth'].cuda(non_blocking=True)

        rgb_image = input_batch['image'].cuda(non_blocking=True)

        if self.use_gravity:
            predicted_normals = self.surface_normal_cnn(rgb_image,
                                                        input_batch['gravity'].cuda(),
                                                        input_batch['aligned_direction'].cuda())
        else:
            predicted_normals = self.surface_normal_cnn(rgb_image)

        if self.args.enriched_samples == 0:
            depth_complete = self.cnn(rgb_image, predicted_normals, ds)
            return depth_complete
        else:
            homo = input_batch['homogeneous_coordinates'].cuda(non_blocking=True)
            di = torch.zeros_like(ds)
            for i in range(ds.shape[0]):
                plane_mask = self.plane_masks_extraction.run_on_tensor(input_batch['image'][i])
                plane_mask = torch.tensor(plane_mask).view(240, 320).cuda(non_blocking=True)
                _, di[i, ...] = extract_plane_images_from_normal_image(predicted_normals[i, ...], plane_mask,
                                                                       ds[i, 0, ...], homo[i, ...])

            # Sample points from each batch of di and add it to ds.
            goal = self.args.enriched_samples
            depth_enriched = ds.clone()
            for batch_id in range(ds.shape[0]):
                msk = di[batch_id, 0, ...] > 0
                nnz = torch.nonzero(msk, as_tuple=True)
                num_select = min(goal, len(nnz[0]))
                sub_indices = np.unique(np.random.randint(0, len(nnz[0]), size=num_select))
                depth_enriched[batch_id, 0, nnz[0][sub_indices], nnz[1][sub_indices]] = di[
                    batch_id, 0, nnz[0][sub_indices], nnz[1][sub_indices]]

            depth_enriched = depth_enriched.cuda(non_blocking=True)
            depth_complete = self.cnn(rgb_image, predicted_normals, depth_enriched)
            return depth_complete


if __name__ == '__main__':
    args = ParseCmdLineArguments()
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    network_run.ConfigureLogging(args.save)

    # First log all the arguments and the values for the record.
    logging.info('sys.argv = {}'.format(sys.argv))
    logging.info('parsed arguments and their values: {}'.format(vars(args)))


    if args.dataset_type == 'scannet':
        train_dataset = ScanNetSmallFramesDataset(usage='train', root=args.root,
                                                  dataset_pickle_file=args.dataset_pickle_file)

        test_dataset = ScanNetSmallFramesDataset(usage='test', root=args.root,
                                                dataset_pickle_file=args.dataset_pickle_file,
                                                skip_every_n_image=args.skip_every_n_image_test)
    elif args.dataset_type == 'nyu':
        train_dataset = NYUDataset(usage='train', root=args.root,
                                  dataset_pickle_file=args.dataset_pickle_file,
                                  use_inpainted_depth=args.use_inpainted_depth)

        test_dataset = NYUDataset(usage='test', root=args.root,
                                  dataset_pickle_file=args.dataset_pickle_file,
                                  skip_every_n_image=args.skip_every_n_image_test,
                                  use_inpainted_depth=args.use_inpainted_depth)
    elif args.dataset_type == 'azure':
        train_dataset = KinectAzureDataset(usage='test',
                                          dataset_pickle_file=args.dataset_pickle_file,
                                          skip_every_n_image=args.skip_every_n_image_test)
        test_dataset = KinectAzureDataset(usage='test',
                                          dataset_pickle_file=args.dataset_pickle_file,
                                          skip_every_n_image=args.skip_every_n_image_test)
    elif args.dataset_type == 'demo':
        train_dataset = DemoDataset(root=args.root)
        test_dataset = DemoDataset(root=args.root)


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.dataloader_train_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.dataloader_test_workers,
                                 pin_memory=True)

    network = RunDepthCompletion(args, train_dataloader, test_dataloader,
                                 network_class_creator=ModifiedFPN, use_gravity=args.use_gravity)

    # Check if this is training or testing.
    if args.train != 0:
        logging.info('Training the network.')
        if args.epoch != 0:
            resume_model = os.path.join(args.save, 'model-epoch-{0:05d}-iter-{1:05d}.ckpt'.format(args.epoch, args.iter))
            network.load_network_from_file(resume_model)
        if args.save == '':
            logging.warning('NO CHECKPOINTS WILL BE SAVED! SET --save FLAG TO SAVE TO A DIRECTORY.')
        network.train(starting_epoch=args.epoch)
    else:
        assert args.checkpoint is not None
        for checkpoint in args.checkpoint:
            network.load_network_from_file(checkpoint)
            network.load_surface_normal_network_from_file(args.surface_normal_checkpoint)
            network.load_plane_extraction_network_from_file(args.plane_detection_config_file)
            network.eval_mode()
            network.evaluate()
