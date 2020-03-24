# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from plane_mask_detection.maskrcnn_benchmark.layers import smooth_l1_loss
from plane_mask_detection.maskrcnn_benchmark.modeling.matcher import Matcher
from plane_mask_detection.maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from plane_mask_detection.maskrcnn_benchmark.modeling.utils import cat

# Add masker to resize mask
from plane_mask_detection.maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_upsampling_grad import MaskerGrad

def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    # FIXME: CPU computation bottleneck, this should be parallelized
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation.
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.masker = MaskerGrad()

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.discretization_size
            )

            labels.append(labels_per_image)
            masks.append(masks_per_image)

        return labels, masks

    def compute_plane_normal_consistency_loss(self, mask_logits, proposals, labels, normal_prediction):
        bbox_index_offsets = [proposals[i].bbox.shape[0] for i in range(len(proposals))]
        bbox_index_offsets.insert(0, 0)
        device = proposals[0].bbox.device
        loss = 0.0
        # print ('proposals len: ', len(proposals))
        # print ('proposals[0]: ', proposals[0])
        # print ('normal_prediction shape: ', normal_prediction.shape)
        W = 0.0
        for i in range(len(proposals)):
            positive_inds = bbox_index_offsets[i] + torch.nonzero(
                labels[bbox_index_offsets[i]:(bbox_index_offsets[i + 1] - 1)] > 0).squeeze(1)
            labels_pos = labels[positive_inds]
            masks_in_ith_image = mask_logits[positive_inds, labels_pos]
            bboxes_in_ith_image = proposals[i]
            # print('is mask still in cuda device?: ', masks_in_ith_image.shape)
            im_masks = self.masker.forward_single_image(masks_in_ith_image, bboxes_in_ith_image)
            # masks_prob = F.relu(im_masks.sigmoid() - 0.5)
            masks_prob = -F.threshold(1e-6 - F.threshold(im_masks, threshold=2.0, value=0.0), threshold=0.0, value=-1.0)
            masks_prob_nograd = masks_prob.detach()
            weights = masks_prob_nograd.sum(dim=(2, 3))
            W += weights.sum()
            # print('weights shape: ', weights.shape)
            # print('masks_prob shape: ', masks_prob.shape)
            for m in range(masks_prob_nograd.shape[0]):
                # print('m: ', m, ', weights[m]: ', weights[m])
                if weights[m] == 0:
                    continue
                weighted_normal_pred = normal_prediction[i].mul(
                    masks_prob_nograd[m, 0].repeat(normal_prediction.shape[1], 1, 1))  # extend to rgb and * with normal pred
                # print('weighted_normal_pred.shape: ', weighted_normal_pred.shape)
                mean_normal = weighted_normal_pred.sum(dim=(1, 2))
                #print('mean_normal after sum: ', mean_normal)
                mean_normal = mean_normal / weights[m]
                # print('mean_normal after div weights: ', mean_normal)
                var_normal = normal_prediction[i] - mean_normal.reshape(normal_prediction.shape[1], 1, 1) \
                    .repeat(1, normal_prediction.shape[2], normal_prediction.shape[3])
                var_normal.mul_(masks_prob_nograd[m, 0].repeat(normal_prediction.shape[1], 1, 1))
                l1_loss = F.l1_loss(var_normal, torch.zeros_like(var_normal), reduction='sum')
                # print('l1_loss: ', l1_loss)
                loss += l1_loss
                #var_normal = var_normal / weights[m]
                #loss += 10000.0 * F.l1_loss(var_normal, torch.zeros_like(var_normal))
            # x = input()
        # var_total /= W
        # loss = F.l1_loss(var_normal, torch.zeros_like(var_normal), reduction='None')
        # print ('W: ', W)
        loss /= W
        loss *= 0.1
        return loss

    def _compute_plane_normal_consistency_loss(self, mask_logits, proposals, labels, normal_prediction):
        bbox_index_offsets = [proposals[i].bbox.shape[0] for i in range(len(proposals))]
        bbox_index_offsets.insert(0, 0)
        device = proposals[0].bbox.device
        loss = 0.0
        # print ('proposals len: ', len(proposals))
        # print ('proposals[0]: ', proposals[0])
        # print ('normal_prediction shape: ', normal_prediction.shape)
        for i in range(len(proposals)):
            positive_inds = bbox_index_offsets[i] + torch.nonzero(
                labels[bbox_index_offsets[i]:(bbox_index_offsets[i + 1] - 1)] > 0).squeeze(1)
            labels_pos = labels[positive_inds]
            masks_in_ith_image = mask_logits[positive_inds, labels_pos]
            bboxes_in_ith_image = proposals[i]
            # print('is mask still in cuda device?: ', masks_in_ith_image.shape)
            im_masks = self.masker.forward_single_image(masks_in_ith_image, bboxes_in_ith_image)
            # masks_prob = F.relu(im_masks.sigmoid() - 0.5)
            masks_prob = -F.threshold(1e-6 - F.threshold(im_masks, threshold=1.39, value=0.0), threshold=0.0, value=-1.0)
            weights = masks_prob.sum(dim=(2, 3))
            # print('weights shape: ', weights.shape)
            # print('masks_prob shape: ', masks_prob.shape)
            for m in range(masks_prob.shape[0]):
                print('m: ', m, ', weights[m]: ', weights[m])
                if weights[m] == 0:
                    continue
                weighted_normal_pred = normal_prediction[i].mul(
                    masks_prob[m, 0].repeat(normal_prediction.shape[1], 1, 1))  # extend to rgb and * with normal pred
                # print('weighted_normal_pred.shape: ', weighted_normal_pred.shape)
                mean_normal = weighted_normal_pred.sum(dim=(1, 2))
                #print('mean_normal after sum: ', mean_normal)
                mean_normal = mean_normal / weights[m]
                # print('mean_normal after div weights: ', mean_normal)
                var_normal = normal_prediction[i] - mean_normal.reshape(normal_prediction.shape[1], 1, 1) \
                    .repeat(1, normal_prediction.shape[2], normal_prediction.shape[3])
                var_normal.mul_(masks_prob[m, 0].repeat(normal_prediction.shape[1], 1, 1))
                l1_loss = F.l1_loss(var_normal, torch.zeros_like(var_normal), reduction='sum')
                print(l1_loss)
                var_normal = var_normal / weights[m]
                # W += weights[m]
                # loss += F.l1_loss(var_normal, torch.zeros_like(var_normal), reduction='None')
                loss += 10000.0 * F.l1_loss(var_normal, torch.zeros_like(var_normal))
            # x = input()
        # var_total /= W
        # loss = F.l1_loss(var_normal, torch.zeros_like(var_normal), reduction='None')
        # loss /= W
        print (loss)
        return loss

    def __call__(self, proposals, mask_logits, targets, normal_prediction=None):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, mask_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[positive_inds, labels_pos], mask_targets
        )
        if normal_prediction is None:
            return mask_loss

        # Compute unsupervised plane-normal consistency loss
        plane_normal_consistency_loss = self.compute_plane_normal_consistency_loss(
            mask_logits, proposals, labels, normal_prediction
        )

        return mask_loss, plane_normal_consistency_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
    )

    return loss_evaluator
