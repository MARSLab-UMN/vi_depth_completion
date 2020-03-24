# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from torch import nn
from plane_mask_detection.maskrcnn_benchmark.layers.misc import interpolate

from plane_mask_detection.maskrcnn_benchmark.structures.bounding_box import BoxList

# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily gor paste_mask_in_image
def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))

    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    # Need to work on the CPU, where fp16 isn't supported - cast to float to avoid this
    device = box.device
    mask = mask.float()
    # box = box.float()

    # padded_mask, scale = expand_masks(mask[None], padding=padding)
    # mask = padded_mask[0, 0]
    # box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)

    TO_REMOVE = 0
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    # print('is mask still in cuda device?: ', mask)
    # print('mask shape before expand: ', mask.shape)
    mask = mask.expand(1, 1, -1, -1)
    # print('mask shape after expand: ', mask.shape)

    # Resize mask
    # mask = mask.to(torch.float32)
    # print('mask shape after to float: ', mask.shape)
    mask = interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    # print('mask shape after interpolate: ', mask.shape)
    mask = mask[0][0]

    # if thresh >= 0:
    #     mask = mask > thresh
    # else:
    #     # for visualization and debugging, we also
    #     # allow it to return an unmodified mask
    #     mask = (mask * 255).to(torch.uint8)


    im_mask = torch.zeros((im_h, im_w), dtype=torch.float, device=device)
    x_0 = max(box[0], 0)
    x_1 = min(box[2], im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3], im_h)

    # print('mask shape after expand: ', mask)
    im_mask[y_0:y_1, x_0:x_1] = mask[
                                        (y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])
                                    ]
    # print('im mask shape after expand: ', mask[(y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])])
    # im_mask[y_0:(y_0+h), x_0:(x_0+w)] = mask
    return im_mask


class MaskerGrad(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert("xyxy")
        im_w, im_h = boxes.size
        res = [
            paste_mask_in_image(mask, box, im_h, im_w, self.threshold, self.padding)
            for mask, box in zip(masks, boxes.bbox)
        ]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]

        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."

        # TODO:  Is this JIT compatible?
        # If not we should make it compatible.
        results = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), "Number of objects should be the same."
            result = self.forward_single_image(mask, box)
            results.append(result)
        return results