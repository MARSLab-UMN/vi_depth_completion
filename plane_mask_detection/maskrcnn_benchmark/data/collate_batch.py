# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from pytorch_local.maskrcnn_benchmark.structures.image_list import to_image_list
from pytorch_local.maskrcnn_benchmark.structures.image_list import to_gravity_tensor


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        normal_images = to_image_list(transposed_batch[2], self.size_divisible)
        mask_images = to_image_list(transposed_batch[3], self.size_divisible)
        img_ids = transposed_batch[4]
        gravity_dirs = None
        gravity_mask = None
        if len(transposed_batch) >= 6:
            gravity_dirs = to_gravity_tensor(transposed_batch[5])
            gravity_mask = to_image_list(transposed_batch[6], self.size_divisible)
        return images, targets, normal_images, mask_images, img_ids, gravity_dirs, gravity_mask


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))

