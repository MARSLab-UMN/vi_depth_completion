# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from plane_mask_detection.maskrcnn_benchmark.structures.image_list import to_image_list
from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..upconv.UpConvNet import UpConvNet
from ..upconv.GravityUpConvNet import GravityUpConvNet
from ..upconv.ResGravityUpConvNet import ResGravityUpConvNet
from ..upconv.UpConvNet import init_weights
from ..upconv.dorn import DORN
import logging
# from ..upconv.dorn import init_weights
import math

### DEBUGGING
import numpy as np
import cv2
import os


from plane_mask_detection.maskrcnn_benchmark.utils.model_serialization import load_state_dict


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        # Transpose convolution for surface normal estimation
        self.device = torch.device(cfg.MODEL.DEVICE)

        logger = logging.getLogger("maskrcnn_benchmark.plane_mask_detection.engine.generalized_rcnn")
        # logger.info('Use MODEL.UPCONV as %s' % cfg.MODEL.UPCONV)
        self.use_gravity = 0
        self.use_gravity_mask = 0
        if cfg.MODEL.UPCONV == "PanopticFPN":
            self.upconv = UpConvNet(cfg)
        elif cfg.MODEL.UPCONV == "GravityPanopticFPN":
            self.upconv = GravityUpConvNet(cfg)
            self.use_gravity = 1
            self.use_gravity_mask = 1
        elif cfg.MODEL.UPCONV == "ResGravityPanopticFPN":
            self.upconv = ResGravityUpConvNet(cfg)
            self.use_gravity = 1
            self.use_gravity_mask = 1
        elif cfg.MODEL.UPCONV == "DORN":
            self.upconv = DORN(cfg)
        else:
            print('generalized rcnn error: cfg.MODEL.UPCONV %s is not implemented' % cfg.MODEL.UPCONV)
        self.upconv.apply(init_weights)

        ######################## TO LOAD NORMAL ONLY STATE DICT: #####################################
        # checkpoint_model = torch.load('/mars/home/tiendo/Code/MASK_RCNN/PanopticFPN/checkpoints_output5/model_0125000.pth',
        #                               map_location=torch.device("cpu"))
        # load_state_dict(self, checkpoint_model.pop("model"))
        ##############################################################################################


        if cfg.SOLVER.SUPERVISED == "plane_normal":
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
            self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-06)
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        self.unsupervised_plane_normal = cfg.SOLVER.UNSUPERVISED.PLANE_NORMAL
        self.supervised_training_scheme = cfg.SOLVER.SUPERVISED
        self.geometric_output = cfg.SOLVER.OUTPUT_REPRESENTATION
        self.type_angle_test = cfg.TEST.EVALUATE_ANGLE

    def forward(self, images, targets=None, normal_images=None, mask_images=None, gravity_dirs=None, gravity_masks=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        features = self.backbone(images.tensors)
        normal_prediction = []
        if self.use_gravity:
            normal_prediction = self.upconv(features, gravity_dirs)
        else:
            normal_prediction = self.upconv(features)
        result = []
        detector_losses = {}
        if self.supervised_training_scheme == "plane_normal":
            proposals, proposal_losses = self.rpn(images, features, targets)
            if self.roi_heads:
                if self.unsupervised_plane_normal:
                    x, result, detector_losses = self.roi_heads(features, proposals, targets, normal_prediction)
                else:
                    # print('in generalized_rcnn:', proposals[0])
                    x, result, detector_losses = self.roi_heads(features, proposals, targets)
            else:
                # RPN-only models don't have roi_heads
                x = features
                result = proposals
                detector_losses = {}

        loss_normal = 0.0
        loss_gravity_aligned = 0.0
        loss_gravity_perp = 0.0
        ch = normal_prediction.shape[1]
        valid_mask = []
        if self.training:
            if self.geometric_output == 0:
                _normal_pred = normal_prediction.permute(0, 2, 3, 1).contiguous().view(-1, ch)
                _gt = normal_images.tensors.permute(0, 2, 3, 1).contiguous().view(-1, ch)
                valid_mask = mask_images.tensors[:, 0] > 0.
                _gt_dot_pred_normalize = self.cosine_similarity(_gt[valid_mask.contiguous().view(-1), :],
                                                                _normal_pred[valid_mask.contiguous().view(-1), :])
                loss_normal = (1.0 - _gt_dot_pred_normalize).mean()
                if self.use_gravity_mask:
                    if torch.sum(gravity_masks.tensors > 0) > 0:
                        _gravity_image = torch.nn.functional.upsample_nearest(gravity_dirs, size=(normal_images.tensors.shape[2],
                                                                                                  normal_images.tensors.shape[3]))
                        _gravity_image = _gravity_image.permute(0, 2, 3, 1).contiguous().view(-1, 3)
                        _gravity_aligned_mask = gravity_masks.tensors[:, 0] == 255
                        _gravity_algined = self.cosine_similarity(_normal_pred[_gravity_aligned_mask.contiguous().view(-1), :],
                                                                    _gravity_image[_gravity_aligned_mask.contiguous().view(-1), :])
                        weight_factor = torch.sum(_gravity_aligned_mask) / torch.tensor(_gravity_aligned_mask.numel(), dtype=torch.float)
                        loss_gravity_aligned += weight_factor * (1.0 - _gravity_algined).mean()
                        # _gravity_perp_mask = gravity_masks.tensors[:, 0] == 127
                        # _gravity_perp = self.cosine_similarity(_normal_pred[_gravity_perp_mask.contiguous().view(-1), :],
                        #                                             _gravity_image[_gravity_perp_mask.contiguous().view(-1), :])
                        # loss_gravity_perp += _gravity_perp.abs().mean()
            elif self.geometric_output == 1:
                ## PLEASE TEST THIS CODE !!!
                ## RE-READ COSINE-SIMILARITY !!!
                # 1. [cosine_phi * cosine_tt <--- axis 2 - x or r]
                # 2. [cosine_phi * sine_tt   <--- axis 1 - y or g]
                # 3. [sine_phi               <--- axis 0 - z or b]
                # 4. Stack these tensors to a new one which is actual prediction
                # 5. Comparison!!!
                tt_prediction = normal_prediction[:, 0, :, :]
                phi_prediction = (normal_prediction[:, 1, :, :] + 1.0) * math.pi/4.0

                ## NOTE: the normal_images is in terms of bgr or zyx
                _gt = nn.functional.normalize(normal_images.tensors, dim=1)

                ## ScanNet
                # valid_mask = normal_images.tensors[:, 0, :, :] > 1e-3
                # valid_mask = valid_mask.detach()
                ## FrameNet
                valid_mask = mask_images.tensors[:, 0] > 0.

                # Put an if here to check if somehow both of these sine and cosine are zeros
                ############################ PREVIOUS LOSS FOR L1 ANGLE ############################
                # gt_tt = torch.atan2(_gt[:, 1, :, :], _gt[:, 2, :, :])
                # gt_phi = torch.atan2(_gt[:, 0, :, :], torch.sqrt(_gt[:, 2, :, :] * _gt[:, 2, :, :]
                #                                                 + _gt[:, 1, :, :] * _gt[:, 1, :, :]))
                # loss_normal = self.normal_loss(tt_prediction[valid_mask], gt_tt[valid_mask])
                # loss_normal += self.normal_loss(phi_prediction[valid_mask], gt_phi[valid_mask])
                #####################################################################################

                gt_tt = torch.atan2(_gt[:, 1, :, :], _gt[:, 2, :, :])
                gt_phi = torch.atan2(_gt[:, 0, :, :], torch.sqrt(_gt[:, 2, :, :] * _gt[:, 2, :, :]
                                                                + _gt[:, 1, :, :] * _gt[:, 1, :, :]))
                # gt_sttctt = nn.functional.normalize(_gt[:, 1:3, :, :], dim=1)
                stt_prediction = torch.sin(tt_prediction)
                stt_gt = torch.sin(gt_tt)
                ctt_prediction = torch.cos(tt_prediction)
                ctt_gt = torch.cos(gt_tt)
                loss_normal = 1.5 * self.l1_loss(stt_prediction[valid_mask], stt_gt[valid_mask])
                loss_normal += 1.5 * self.l1_loss(ctt_prediction[valid_mask], ctt_gt[valid_mask])
                loss_normal += self.l1_loss(phi_prediction[valid_mask], gt_phi[valid_mask])
            elif self.geometric_output == 2 or self.geometric_output == 3:
                _normal_pred = normal_prediction.permute(0, 2, 3, 1).contiguous().view(-1, ch)
                _normal_pred[:, 0] = 0.5 * (_normal_pred[:, 0] + 1.0)
                _gt = normal_images.tensors.permute(0, 2, 3, 1).contiguous().view(-1, ch)
                ## ScanNet
                # valid_mask = _gt[:, 0] > 1e-3
                # valid_mask = valid_mask.detach()
                ## FrameNet
                valid_mask = mask_images.tensors[:, 0] > 0.
                _gt_dot_pred_normalize = self.cosine_similarity(_gt[valid_mask.contiguous().view(-1), :],
                                                                _normal_pred[valid_mask.contiguous().view(-1), :])
                target_ones = torch.ones(_gt_dot_pred_normalize.shape[0]).to(self.device)
                loss_normal = (target_ones - _gt_dot_pred_normalize).mean()
            elif self.geometric_output == 4:
                tt_prediction = normal_prediction[:, 0, :, :] * math.pi
                phi_prediction = (normal_prediction[:, 1, :, :] + 1.0) * math.pi / 4.0

                ## NOTE: the normal_images is in terms of bgr or zyx
                _gt = nn.functional.normalize(normal_images.tensors, dim=1)

                ## ScanNet
                # valid_mask = normal_images.tensors[:, 0, :, :] > 1e-3
                # valid_mask = valid_mask.detach()
                ## FrameNet
                valid_mask = mask_images.tensors[:, 0] > 0.

                gt_tt = torch.atan2(_gt[:, 1, :, :], _gt[:, 2, :, :])
                gt_phi = torch.atan2(_gt[:, 0, :, :], torch.sqrt(_gt[:, 2, :, :] * _gt[:, 2, :, :]
                                                                + _gt[:, 1, :, :] * _gt[:, 1, :, :]))
                loss_normal = self.l1_loss(tt_prediction[valid_mask], gt_tt[valid_mask])
                loss_normal += self.l1_loss(phi_prediction[valid_mask], gt_phi[valid_mask])
            elif self.geometric_output == 5:
                sttctt_prediction = nn.functional.normalize(normal_prediction[:, 0:2, :, :], dim=1)
                phi_prediction = (normal_prediction[:, 2, :, :] + 1.0) * math.pi / 4.0
                ## NOTE: the normal_images is in terms of bgr or zyx
                _gt = nn.functional.normalize(normal_images.tensors, dim=1)
                ## ScanNet
                # valid_mask = normal_images.tensors[:, 0, :, :] > 1e-3
                # valid_mask = valid_mask.detach()
                ## FrameNet
                valid_mask = mask_images.tensors[:, 0] > 0.
                # Put an if here to check if somehow both of these sine and cosine are zeros
                gt_phi = torch.atan2(_gt[:, 0, :, :], torch.sqrt(_gt[:, 2, :, :] * _gt[:, 2, :, :]
                                                                 + _gt[:, 1, :, :] * _gt[:, 1, :, :]))
                gt_sttctt = nn.functional.normalize(_gt[:, 1:3, :, :], dim=1)
                loss_normal = 1.5*self.l1_loss(sttctt_prediction[:, 0, :, :][valid_mask], gt_sttctt[:, 0, :, :][valid_mask])
                loss_normal += 1.5*self.l1_loss(sttctt_prediction[:, 1, :, :][valid_mask],
                                            gt_sttctt[:, 1, :, :][valid_mask])
                loss_normal += self.l1_loss(phi_prediction[valid_mask], gt_phi[valid_mask])
            elif self.geometric_output == 6:
                _normal_pred = torch.nn.functional.normalize(normal_prediction.permute(0, 2, 3, 1).contiguous().view(-1, ch), dim=1)
                _gt = torch.nn.functional.normalize(normal_images.tensors.permute(0, 2, 3, 1).contiguous().view(-1, ch), dim=1)

                ## ScanNet
                # valid_mask = _gt[:, 0] > 1e-3
                # valid_mask = valid_mask.detach()
                ## FrameNet
                valid_mask = mask_images.tensors[:, 0] > 0.
                loss_normal = self.l1_loss(_gt[valid_mask.contiguous().view(-1), :],
                                           _normal_pred[valid_mask.contiguous().view(-1), :])
        else: # Testing/Evaluation/Inference
            return result, None, None
            _gt = normal_images.tensors.permute(0, 2, 3, 1).contiguous().view(normal_images.tensors.shape[0],
                                                                              normal_images.tensors.shape[2] *
                                                                              normal_images.tensors.shape[3],
                                                                              normal_images.tensors.shape[1])
            valid_mask = [mask_images.tensors[i, 0].contiguous().view(-1) > 0. for i in
                          range(normal_prediction.shape[0])]

            if self.geometric_output == 1 or self.geometric_output == 4:
                normal_prediction[:, 1] += 1.0
                normal_prediction[:, 1] *= math.pi/4.0
                if self.geometric_output == 4:
                    normal_prediction[:, 0] *= math.pi

                n = [torch.stack((torch.sin(normal_prediction[i, 1]),
                                  torch.sin(normal_prediction[i, 0])*torch.cos(normal_prediction[i, 1]),
                                  torch.cos(normal_prediction[i, 0])*torch.cos(normal_prediction[i, 1]),
                                  ), dim=2).contiguous().view(-1, 3)
                                for i in range(normal_prediction.shape[0])]
            elif self.geometric_output == 5:
                normal_prediction[:, 2] += 1.0
                normal_prediction[:, 2] *= math.pi/4.0

                sttctt_prediction = nn.functional.normalize(normal_prediction[:, 0:2, :, :], dim=1)

                n = [torch.stack((torch.sin(normal_prediction[i, 2]),
                                  sttctt_prediction[i, 0]*torch.cos(normal_prediction[i, 2]),
                                  sttctt_prediction[i, 1]*torch.cos(normal_prediction[i, 2]),
                                  ), dim=2).contiguous().view(-1, 3)
                                for i in range(normal_prediction.shape[0])]
            else:
                # normal_prediction[:, 0] = 0.5*(normal_prediction[:, 0] + 1.0)
                n = [torch.nn.functional.normalize(normal_prediction[i].permute(1, 2, 0).contiguous().view(-1, 3), dim=1)
                            for i in range(normal_prediction.shape[0])]

        # # DEBUG for mask predictions
        # rgb_img = np.zeros((240, 320, 3), dtype=float)
        # rgb_img = np.asarray(rgb_img, dtype=float)
        # pil_images = images.tensors.cpu()
        # # print('proposals: ', proposals[0])
        # # # print('images:', images.tensors.shape)
        # # # print('image tensor: ', pil_images.shape)
        # # print('targets: ', targets[0])
        # np_img = np.array(pil_images[0].permute(1, 2, 0))
        # if not cv2.imwrite('/mars/home/tiendo/Code/MASK_RCNN/PanopticFPN/plane_normal/mask_gt.jpg', np_img):
        #     print('saving failed')
        # bbox_cpu = targets[0].bbox.cpu()
        # proposal_bbox_cpu = proposals[0].bbox.cpu()
        # # print (bbox_cpu)
        # # print (proposal_bbox_cpu)
        # for i in range(len(proposal_bbox_cpu)):
        #     bbox = np.asarray(proposal_bbox_cpu[i])
        #     bbox = bbox.astype(dtype=np.int)
        #     # print('bbox: ', proposal_bbox_cpu[i])
        #     rgb_img = cv2.rectangle(rgb_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))
        #
        # if not cv2.imwrite('/mars/home/tiendo/Code/MASK_RCNN/PanopticFPN/plane_normal/mask_pred.jpg', rgb_img):
        #     print('saving failed')
        # # #
        # for i in range(len(bbox_cpu)):
        #     bbox = np.asarray(bbox_cpu[i])
        #     bbox = bbox.astype(dtype=np.int)
        # # #     # print('bbox: ', bbox_cpu[i])
        # #     bbox_img = np_img
        #     count_proposal_found = 0
        #     for j in range(len(proposal_bbox_cpu)):
        #         pbbox = np.asarray(proposal_bbox_cpu[j])
        #         pbbox = pbbox.astype(dtype=np.int)
        # #         # print('bbox: ', proposal_bbox_cpu[j])
        #         iou_score = bb_intersection_over_union(bbox, pbbox)
        #         if iou_score > 0.7:
        #             # bbox_img = cv2.rectangle(bbox_img, (pbbox[0], pbbox[1]), (pbbox[2], pbbox[3]), (0, 0, 255), thickness = 3)
        #             count_proposal_found += 1
        #     print('count_proposal_found: ', count_proposal_found)
        # #     bbox_img = cv2.rectangle(bbox_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), thickness=5)
        # # #     cv2.imshow('img', bbox_img)
        # # #     cv2.waitKey(0)

        if self.training:
            losses = {}
            if self.use_gravity:
                losses.update(dict(loss_gravity_aligned=loss_gravity_aligned))
                losses.update(dict(loss_gravity_perp=loss_gravity_perp))
            if self.supervised_training_scheme == 'normal_only':
                losses.update(dict(loss_normal=loss_normal))
                return losses
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(dict(loss_normal=loss_normal))
            return losses
        else:
        # In testing, we extract delta, slant, tilt error statistics as well as the visualization of
        # the normal estimation error mask
            if self.type_angle_test == "delta":
                error_normal = [self.cosine_similarity(_gt[i, valid_mask[i], :], n[i][valid_mask[i], :]) for i in
                                range(normal_prediction.shape[0])]
            elif self.type_angle_test == "slant":
                gt_phi = torch.atan2(_gt[:, :, 0], torch.sqrt(_gt[:, :, 2] * _gt[:, :, 2]
                                                              + _gt[:, :, 1] * _gt[:, :, 1]))
                error_normal = [torch.cos(gt_phi[i, valid_mask[i]] -
                                          torch.atan2(n[i][valid_mask[i], 0],
                                                      torch.sqrt(n[i][valid_mask[i], 1] * n[i][valid_mask[i], 1]
                                                                 + n[i][valid_mask[i], 2] * n[i][valid_mask[i], 2])))
                                for i in range(normal_prediction.shape[0])]
            elif self.type_angle_test == "tilt":
                gt_tt = torch.atan2(_gt[:, :, 1], _gt[:, :, 2])
                error_normal = [torch.cos(gt_tt[i, valid_mask[i]] - torch.atan2(n[i][valid_mask[i], 1], n[i][valid_mask[i], 2]))
                                for i in range(normal_prediction.shape[0])]

        return result, error_normal, n


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
