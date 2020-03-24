# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F

import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, normal_image=None, mask_image=None, gravity_mask=None):
        for t in self.transforms:
            if normal_image is not None:
                if gravity_mask is not None:
                    image, target, normal_image, mask_image, gravity_mask = t(image, target, normal_image, mask_image, gravity_mask)
                else:
                    image, target, normal_image, mask_image = t(image, target, normal_image, mask_image)
            else:
                image, target = t(image, target)
        if normal_image is not None:
            if gravity_mask is not None:
                return image, target, normal_image, mask_image, gravity_mask
            else:
                return image, target, normal_image, mask_image
        else:
            return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None, normal_image=None, mask_image=None, gravity_mask=None):
        """
        NOTE: The resize here is forced to be min and max size
        It closely approximate the aspect ratio (but not exactly the same as original image)
        at the same time, the image size is cut twice and the resize image is divisible by 64
        """
        # size = self.get_size(image.size)
        size = (max(self.min_size), self.max_size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        if normal_image is not None:
            normal_image = F.resize(normal_image, size)
            mask_image = F.resize(mask_image, size)
            if gravity_mask is not None:
                gravity_mask = F.resize(gravity_mask, size)
                return image, target, normal_image, mask_image, gravity_mask
            else:
                return image, target, normal_image, mask_image
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target, normal_image=None, mask_image=None, gravity_mask=None):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
            if normal_image is not None:
                normal_image = F.hflip(normal_image)
                mask_image = F.hflip(mask_image)
            if gravity_mask is not None:
                gravity_mask = F.hflip(gravity_mask)
        if normal_image is not None:
            if gravity_mask is not None:
                return image, target, normal_image, mask_image, gravity_mask
            else:
                return image, target, normal_image, mask_image
        else:
            return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target, normal_image=None, mask_image=None, gravity_mask=None):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
            if normal_image is not None:
                normal_image = F.vflip(normal_image)
                mask_image = F.vflip(mask_image)
            if gravity_mask is not None:
                gravity_mask = F.vlip(gravity_mask)
        if normal_image is not None:
            if gravity_mask is not None:
                return image, target, normal_image, mask_image, gravity_mask
            else:
                return image, target, normal_image, mask_image
        else:
            return image, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target, normal_image=None, mask_image=None, gravity_mask=None):
        image = self.color_jitter(image)
        if normal_image is not None:
            if gravity_mask is not None:
                return image, target, normal_image, mask_image, gravity_mask
            else:
                return image, target, normal_image, mask_image
        else:
            return image, target


class ToTensor(object):
    def __call__(self, image, target, normal_image=None, mask_image=None, gravity_mask=None):
        if normal_image is not None:
            if gravity_mask is not None:
                return F.to_tensor(image), target, F.to_tensor(normal_image), F.to_tensor(mask_image), F.to_tensor(gravity_mask)
            else:
                return F.to_tensor(image), target, F.to_tensor(normal_image), F.to_tensor(mask_image)
        else:
            return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None, normal_image=None, mask_image=None, gravity_mask=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        if normal_image is not None:
            normal_image = normal_image[[2, 1, 0]]
            normal_image = F.normalize(-normal_image, mean=[-0.5, -0.5, -0.5], std=self.std)
            mask_image = (mask_image > 0)
        if normal_image is not None:
            if gravity_mask is not None:
                gravity_mask *= 255
                return image, target, normal_image, mask_image, gravity_mask
            else:
                return image, target, normal_image, mask_image
        else:
            return image, target
