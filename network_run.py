import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms.functional as TVF
import numpy as np
from PIL import Image

import logging
import os
import sys
from datetime import datetime

import normal_utils

def ConfigureLogging(save_path):
    if save_path != '':
        filename = 'log_' + sys.argv[0] + datetime.now().strftime('_%Y%m%d_%H%M%S') + '.log'
        os.makedirs(save_path, exist_ok=True)
        full_file = os.path.join(save_path, filename)
        handlers = [logging.StreamHandler(), logging.FileHandler(full_file)]
    else:
        handlers = [logging.StreamHandler()]

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-1.1s%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d] %(message)s",
        datefmt='%m%d %H:%M:%S',
        handlers=handlers)


def SaveNormalsToImage(normals, filename):
    assert(len(normals.shape) == 3)
    assert(normals.shape[0] == 3)
    normals = normals.transpose([1, 2, 0])
    # Save in FrameNet format.
    normals = (1 + normals) * 127.5
    image = Image.fromarray(normals.astype(np.uint8))
    image.save(filename)


def SaveDepthsToImage(depths, filename):
    if len(depths.shape) == 3:
        assert(depths.shape[0] == 1)
        depths = depths.squeeze()
    assert(len(depths.shape) == 2)
    # The PNG format sometimes does not support writing to uint16 image in Pillow package, so
    # we save in uint32 format.
    image = Image.fromarray((depths * 1000).astype(np.uint32))
    image.save(filename)


def SaveMasksToImage(mask, filename):
    if len(mask.shape) == 3:
        assert(mask.shape[0] == 1)
        mask = mask.squeeze()
    assert(len(mask.shape) == 2)
    image = Image.fromarray(mask)
    image.save(filename)


def SaveRgbToImage(rgb, filename):
    if len(rgb.shape) == 4:
        assert rgb.shape[0] == 1
        rgb =rgb.squeeze()
    assert len(rgb.shape) == 3
    np_image = np.transpose(rgb * 255, axes=[1, 2, 0]).astype(np.uint8)
    image = TVF.to_pil_image(np_image, mode='RGB')
    image.save(filename)


def GetDepthPrintableRatios(valid_gt, valid_preds):
    ratios = torch.max(valid_preds / valid_gt, valid_gt / valid_preds)

    r05 = 100 * torch.sum(ratios < 1.05).item() / (ratios.nelement() + 1e-7)
    r10 = 100 * torch.sum(ratios < 1.10).item() / (ratios.nelement() + 1e-7)
    r25 = 100 * torch.sum(ratios < 1.25).item() / (ratios.nelement() + 1e-7)
    r25_2 = 100 * torch.sum(ratios < 1.25**2).item() / (ratios.nelement() + 1e-7)
    r25_3 = 100 * torch.sum(ratios < 1.25**3).item() / (ratios.nelement() + 1e-7)

    return {'D_1.05': round(r05, 1), 'D_1.10': round(r10, 1), 'D_1.25': round(r25, 1),
            'D_1.56': round(r25_2, 1), 'D_1.95': round(r25_3, 1)}


# NOTE: This network is primarily used to estimate depth and/or normal.
class ImageNetworkRunInterface:
    def __init__(self, arguments, train_dataloader, test_dataloader):
        self.args = arguments
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        if self.args.save != '':
            if not os.path.exists(self.args.save):
                os.mkdir(self.args.save)

        self.cnn = self._prepare_network()
        if self.args.enable_multi_gpu != 0 and torch.cuda.device_count() > 1:
            logging.info('Using {0} GPUs.'.format(torch.cuda.device_count()))
            self.cnn = torch.nn.DataParallel(self.cnn)
        self.cnn = self.cnn.cuda()
        self.optimizer = self._create_optimizer(self.cnn, arguments.learning_rate)

    #
    # Interface functions that should be defined by the subclass.
    #
    def _on_new_epoch(self, epoch):
        pass

    # Called at the start of each iteration.
    def _on_new_iteration(self, epoch, iteration):
        pass

    # Called right after the network output is produced. Useful for debugging.
    def _on_network_output(self, epoch, iteration, input_batch, network_output):
        pass

    # Creates and prepares the network. Gets called at construction.
    def _prepare_network(self):
        return None

    # Determines if the network will estimate the normal vectors.
    def _network_estimates_normal(self):
        return False

    # Determines if the network will estimate the depth values.
    def _network_estimates_depth(self):
        return False

    # If the network estimates the depth, this function will receive the network
    # output (which can be a tuple) and returns the depth part of it.
    # NOTE: If the network estimates normal or returns multiple outputs, this
    # function must be overridden in the subclass.
    def _get_network_output_depth(self, network_output):
        assert(not self._network_estimates_normal())
        assert(self._network_estimates_depth())
        return network_output

    # If the network estimates the normal, this function receives the network
    # output and returns the normal part of it.
    # NOTE: If the network estimates depth or returns multiple outputs, this
    # function must be overridden in the subclass.
    def _get_network_output_normal(self, network_output):
        assert(self._network_estimates_normal())
        assert(not self._network_estimates_depth())
        return network_output

    # Calls the network and returns its results. There is no need to re-define this
    # function if the input to the network is only the images.
    def _call_cnn(self, input_batch):
        images = Variable(input_batch['image']).cuda()
        return self.cnn(images)

    # Given the network output and the input images and ground truth, compute and return the loss.
    # The second return parameter can be either None, or any other value that the network want to
    # send to the output (not used for optimization).
    # The default implementation only checks if the network has normal and/or depth losses and returns.
    # The implementation can modify this behavior in the subclass.
    def _network_loss(self, input_batch, cnn_outputs):
        losses_map = {}
        other_outputs = {}

        _, _, height, width = input_batch['image'].shape
        image_size = height * width

        if self._network_estimates_depth():
            depths_gt = input_batch['depth'].float().cuda()
            depth_mask = depths_gt > 0

            depth_loss_func = torch.nn.L1Loss(reduction='sum')
            pred_depth = self._get_network_output_depth(cnn_outputs)
            depth_loss = depth_loss_func(pred_depth[depth_mask], depths_gt[depth_mask])
            losses_map['depth_L1'] = depth_loss / image_size

            # Compute the ratios.
            valid_preds = pred_depth[depth_mask]
            valid_gt = depths_gt[depth_mask]

            printable_ratios = GetDepthPrintableRatios(valid_gt, valid_preds)
            other_outputs.update(printable_ratios)

        if self._network_estimates_normal():
            mask = (input_batch['mask'].cuda() > 0)[:, None, :, :]
            norm_gt = F.normalize(input_batch['normal'].cuda())

            pred_normals = self._get_network_output_normal(cnn_outputs)
            angle_loss, angle = normal_utils.compute_normal_vectors_loss_l1(norm_gt, pred_normals, mask)
            loss_name = 'angle_L1'
            losses_map[loss_name] = angle_loss
            other_outputs['angle'] = round(angle.item(), 4)

        return losses_map, other_outputs

    # Given the network output and the ground-truth inputs, evaluate the network errors.
    # Note that this function will always return two parameters. The first one is the normal angle error,
    # while the second one is the depth error. If the network does not estimate one or the other, the value
    # of the returned corresponding variable will be None. Furthermore, both variables should be either None
    # or numpy arrays.
    def _network_evaluate(self, input_batch, cnn_outputs):
        normal_error = None
        depth_ratio_error = None
        depth_abs_error = None


        if self._network_estimates_normal():
            pred_normals = F.normalize(self._get_network_output_normal(cnn_outputs))
            mask = (input_batch['mask'].cuda() > 0)[:, None, :, :]
            norm_gt = F.normalize(input_batch['normal'].cuda())

            dot_product = torch.sum(pred_normals * norm_gt, dim=1)
            dot_product = torch.clamp(dot_product, min=-1.0, max=1.0)
            angles_error = torch.acos(dot_product) / np.pi * 180
            mask_np = mask[:, 0, :, :].detach().cpu().numpy() > 0
            angles_np = angles_error.detach().cpu().numpy()
            normal_error = angles_np[mask_np]

        if self._network_estimates_depth():
            pred_depths = self._get_network_output_depth(cnn_outputs)
            depths_gt = input_batch['depth'].cuda()
            depth_mask = depths_gt > 0
            depth_mask_np = depth_mask.detach().cpu().numpy() > 0
            depth_ratio_np = torch.max(depths_gt / pred_depths, pred_depths / depths_gt).detach().cpu().numpy()
            depth_ratio_error = depth_ratio_np[depth_mask_np]
            depth_abs_error = (depths_gt - pred_depths).abs().detach().cpu().numpy()[depth_mask_np]

        return normal_error, depth_ratio_error, depth_abs_error

    # Creates the optimizer object. Gets called from the constructor.
    def _create_optimizer(self, cnn, learning_rate):
        return torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    def _run_training_iteration(self, sample_batched, epoch, max_epochs, iteration, max_iters):
        self.cnn.train()
        self.optimizer.zero_grad()

        # This function will call the CNN and return its outputs
        cnn_outputs = self._call_cnn(sample_batched)
        self._on_network_output(epoch, iteration, sample_batched, cnn_outputs)

        losses_map, other_outputs_map = self._network_loss(sample_batched, cnn_outputs)

        # Losses map is a mapping from a string to loss.
        total_loss = 0.0
        name_to_value_map = {}
        for name, loss in losses_map.items():
            total_loss += loss
            name_to_value_map[name] = round(loss.item(), 4)

        total_loss.backward()
        self.optimizer.step()

        logging.info('Epoch {0}/{1}, Iter {2}/{3}. Total loss: {4:.4f}. Breakdown: {5}'.format(
            epoch, max_epochs, iteration, max_iters, total_loss.item(), name_to_value_map))
        if other_outputs_map is not None and len(other_outputs_map) > 0:
            logging.info('{}'.format(other_outputs_map))

    # Override this method to save extra information to the output path in addition to depth/normal
    # prediction and ground-truth images.
    def _save_extra_evaluation_outputs(self, sample_batched, cnn_outputs, start_index, output_path):
        pass

    def _network_save_cnn_evaluation_output(self, sample_batched, cnn_outputs, start_index, output_path):
        if self._network_estimates_normal():
            pred_normals = F.normalize(self._get_network_output_normal(cnn_outputs))
            assert(pred_normals.shape[0] == self.args.batch_size)
            norm_gt = F.normalize(sample_batched['normal'].cuda())

            for i in range(self.args.batch_size):
                SaveNormalsToImage(pred_normals[i, ...].squeeze().detach().cpu().numpy(),
                                   os.path.join(output_path, 'normal_pred_{0:06d}.png'.format(start_index + i)))
                SaveNormalsToImage(norm_gt[i, ...].squeeze().detach().cpu().numpy(),
                                   os.path.join(output_path, 'normal_gt_{0:06d}.png'.format(start_index + i)))
        if self._network_estimates_depth():
            pred_depths = self._get_network_output_depth(cnn_outputs)
            assert(pred_depths.shape[0] <= self.args.batch_size)
            depths_gt = sample_batched['depth'].cuda()

            for i in range(pred_depths.shape[0]):
                SaveDepthsToImage(pred_depths[i, ...].squeeze().detach().cpu().numpy(),
                                  os.path.join(output_path, 'depth_pred_{0:06d}.png'.format(start_index + i)))
                SaveDepthsToImage(depths_gt[i, ...].squeeze().detach().cpu().numpy(),
                                  os.path.join(output_path, 'depth_gt_{0:06d}.png'.format(start_index + i)))
        self._save_extra_evaluation_outputs(sample_batched, cnn_outputs, start_index, output_path)

    def _network_save_cnn_output_pred_only(self, sample_batched, cnn_outputs, output_path):
        if self._network_estimates_depth():
            pred_depths = self._get_network_output_depth(cnn_outputs)
            assert(pred_depths.shape[0] <= self.args.batch_size)

            for i in range(pred_depths.shape[0]):
                file_index = int(sample_batched['color_filename'][i][6:12])
                SaveDepthsToImage(pred_depths[i, ...].squeeze().detach().cpu().numpy(),
                                  os.path.join(output_path, 'depth_pred_{0:06d}.png'.format(file_index)))

    def _run_evaluation_iteration(self, sample_batched, iteration):
        self.cnn.eval()
        cnn_outputs = self._call_cnn(sample_batched)

        # Log the output if provided the path.
        if self.args.save != '':
            start_index = iteration * self.args.batch_size
            output_path = os.path.join(self.args.save, 'evaluation_output')
            os.makedirs(output_path, exist_ok=True)

            ## NOTE: ADDING FOR DEMO PURPOSE
            if self.args.dataset_type == 'demo':
                self._network_save_cnn_output_pred_only(sample_batched, cnn_outputs, output_path)
            else:
                self._network_save_cnn_evaluation_output(sample_batched, cnn_outputs, start_index, output_path)

        self._on_network_output(0, iteration, sample_batched, cnn_outputs)

        if self.args.dataset_type == 'demo':
            angle_errors, depth_rel_errors, depth_abs_errors = None, None, None
        else:
            angle_errors, depth_rel_errors, depth_abs_errors = self._network_evaluate(sample_batched, cnn_outputs)

        return angle_errors, depth_rel_errors, depth_abs_errors

    def load_network_from_file(self, filename):
        logging.info('Loading network from file {}'.format(filename))
        state = self.cnn.state_dict()
        state.update(torch.load(filename))
        self.cnn.load_state_dict(state)

    def train(self, starting_epoch=0, max_epochs=10000):
        max_iters = len(self.train_dataloader.dataset) // self.args.batch_size

        for epoch in range(starting_epoch, max_epochs):
            self._on_new_epoch(epoch)

            for i, sample_batched in enumerate(self.train_dataloader):
                self._on_new_iteration(epoch, i)
                self._run_training_iteration(sample_batched, epoch, max_epochs, i, max_iters)

                if self.args.save != '' and ((i % self.args.save_every_n_iteration == 0 and i > 0) or (i == 0 and epoch % self.args.save_every_n_epoch == 0)):
                    path = os.path.join(self.args.save, 'model-epoch-%05d-iter-%05d.ckpt' % (epoch, i))
                    torch.save(self.cnn.state_dict(), path)

                # Do not test on epoch 0 iteration 0, since it doesn't have much meaning.
                if i % self.args.eval_test_every_n_iterations == 0 and (i != 0 or epoch != 0):
                    logging.info('Evaluating the network on test set.')
                    # Note: this does not save anything.
                    tmp_save = self.args.save
                    self.args.save = ''
                    self.evaluate()
                    self.args.save = tmp_save


    def evaluate(self):
        total_depth_ratio_errors = None
        total_depth_abs_errors = None
        total_normal_errors = None
        if self.args.save != '':
            logging.info('Writing the output normal/depth images to {}.'.format(os.path.join(self.args.save, 'evaluation_output')))
        all_normal_errors = None
        all_depth_ratio_errors = None
        all_depth_absolute_errors = None
        for i, sample_batched in enumerate(self.test_dataloader):
            with torch.no_grad():
                angle_errors, depth_ratio_errors, depth_absolute_errors = self._run_evaluation_iteration(sample_batched, i)

                if angle_errors is not None:
                    if all_normal_errors is None:
                        all_normal_errors = []
                    all_normal_errors.append(angle_errors)
                if depth_ratio_errors is not None:
                    if all_depth_ratio_errors is None:
                        all_depth_ratio_errors = []
                    all_depth_ratio_errors.append(depth_ratio_errors)
                if depth_absolute_errors is not None:
                    if all_depth_absolute_errors is None:
                        all_depth_absolute_errors = []
                    all_depth_absolute_errors.append(depth_absolute_errors)

        total_normal_errors = None
        if all_normal_errors is not None:
            total_normal_errors = np.concatenate(all_normal_errors)

        total_depth_ratio_errors = None
        if all_depth_ratio_errors is not None:
            total_depth_ratio_errors = np.concatenate(all_depth_ratio_errors)

        total_depth_abs_errors = None
        if all_depth_absolute_errors is not None:
            total_depth_abs_errors = np.concatenate(all_depth_absolute_errors)

        if total_normal_errors is not None:
            logging.info('NORMAL ERROR STATS: Mean %f, Median %f, Rmse %f, 5deg %f, 7.5deg %f, 11.25deg %f, 22.5deg %f, 30deg %f' %
                         (np.average(total_normal_errors), np.median(total_normal_errors),
                          np.sqrt(np.sum(total_normal_errors * total_normal_errors) / total_normal_errors.shape),
                          100 * np.sum(total_normal_errors < 5) / total_normal_errors.shape[0],
                          100 * np.sum(total_normal_errors < 7.5) / total_normal_errors.shape[0],
                          100 * np.sum(total_normal_errors < 11.25) / total_normal_errors.shape[0],
                          100 * np.sum(total_normal_errors < 22.5) / total_normal_errors.shape[0],
                          100 * np.sum(total_normal_errors < 30) / total_normal_errors.shape[0]))
        if total_depth_ratio_errors is not None:
            logging.info('DEPTH ERROR STATS: MAD: %f, RMSE: %f, 1.05 %f 1.10: %f 1.25: %f 1.25^2: %f, 1.25^3: %f' %
                         (np.mean(total_depth_abs_errors), np.sqrt(np.mean(total_depth_abs_errors ** 2)),
                          100 * np.sum(total_depth_ratio_errors < 1.05) / total_depth_ratio_errors.shape[0],
                          100 * np.sum(total_depth_ratio_errors < 1.10) / total_depth_ratio_errors.shape[0],
                          100 * np.sum(total_depth_ratio_errors < 1.25) / total_depth_ratio_errors.shape[0],
                          100 * np.sum(total_depth_ratio_errors < 1.25**2) / total_depth_ratio_errors.shape[0],
                          100 * np.sum(total_depth_ratio_errors < 1.25**3) / total_depth_ratio_errors.shape[0]))


# This class creates a network run interface for any CNN that estimates depth and/or normal.
# If the network network does estimate both depth and normal, then it is assumed that the network
# output is in the order of (normal, depth). Furthermore, any additional network outputs are ignored.
# If it only estimates one of the normal or depth, then it assumes that the network outputs only that parameter.
# At construction in addition to the train and test loaders, this class receives the class name of the CNN or
# a function which returns the constructed cnn, and whether the network estimates either of normal or depth
# values from the RGB image (it is assumed that this is the only output the network requires).
class DefaultImageNetwork(ImageNetworkRunInterface):
    def __init__(self, arguments, train_loader, test_loader, network_class,
                 estimates_depth=False, estimates_normal=False):
        self._network_class = network_class
        super(DefaultImageNetwork, self).__init__(arguments, train_loader, test_loader)
        self.estimates_depth = estimates_depth
        self.estimates_normal = estimates_normal
        assert(estimates_depth or estimates_normal)

    def _prepare_network(self):
        net_cls = self._network_class
        return net_cls()

    def _network_estimates_normal(self):
        return self.estimates_normal

    def _network_estimates_depth(self):
        return self.estimates_depth

    def _get_network_output_normal(self, network_output):
        assert(self._network_estimates_normal())
        if self._network_estimates_depth():
            return network_output[0]
        else:
            return network_output

    def _get_network_output_depth(self, network_output):
        assert(self._network_estimates_depth())
        if self._network_estimates_normal():
            return network_output[1]
        else:
            return network_output
