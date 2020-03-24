# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from pytorch_local.maskrcnn_benchmark.structures.bounding_box import BoxList
from pytorch_local.maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from pytorch_local.maskrcnn_benchmark.structures.keypoint import PersonKeypoints

# For debugging
import cv2
import numpy as np
from PIL import Image
import os
import os.path
from torchvision.transforms import functional as F

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, gravity=0
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.use_gravity = gravity

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # print('img coco index: ', idx)
        # print('img id map: ', self.id_to_img_map[idx])
        # print('img: ', img.shape)
        # np_img = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
        # np_img = np_img.astype(dtype=np.uint8)
        # for i in range(len(anno)):
        #     bbox = np.asarray(anno[i]['bbox'])
        #     bbox = bbox.astype(dtype=np.int)
        #     print('bbox: ', anno[i]['bbox'])
        #     print('category_id: ', anno[i]['category_id'])
        #     np_img = cv2.rectangle(np_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0))
        #
        # cv2.imshow('img', np_img)
        # cv2.waitKey(0)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        # Load normal image
        str_image_id = str(self.id_to_img_map[idx])
        normal_img_path = 'scene%04d_%02d/frame-%06d-normal.png' % (int(str_image_id[1:4]),
                                                                    int(str_image_id[4:6]),
                                                                    int(str_image_id[6:12]))
        mask_img_path = 'scene%04d_%02d/frame-%06d-orient-mask.png' % (int(str_image_id[1:4]),
                                                                        int(str_image_id[4:6]),
                                                                        int(str_image_id[6:12]))

        # ### NYUD
        # mask_img_path = "/mars/home/tiendo/Code/nyud_v2/mask/frame-%06d-mask.png" % (int(str_image_id[6:12]))
        # mask_img = Image.open(mask_img_path)
        # #####



        normal_img = Image.open(os.path.join(self.root, normal_img_path)).convert('RGB')
        mask_img = Image.open(os.path.join(self.root, mask_img_path))
        if self.use_gravity:
            gravity_dir_path = 'scene%04d_%02d/frame-%06d-gravity.txt' % (int(str_image_id[1:4]),
                                                                          int(str_image_id[4:6]),
                                                                          int(str_image_id[6:12]))
            # gravity_dir_path = 'scene%04d_%02d/frame-%06d-plane_normal_gravity.txt' % (int(str_image_id[1:4]),
            #                                                                            int(str_image_id[4:6]),
            #                                                                            int(str_image_id[6:12]))
            gravity_mask_path = 'scene%04d_%02d/frame-%06d-gravity-mask.png' % (int(str_image_id[1:4]),
                                                                                int(str_image_id[4:6]),
                                                                                int(str_image_id[6:12]))
            gravity_dir = torch.tensor(np.loadtxt(os.path.join(self.root, gravity_dir_path), dtype=np.float), dtype=torch.float)
            gravity_dir = torch.tensor([gravity_dir[2], gravity_dir[1], gravity_dir[0]])
            gravity_mask = Image.open(os.path.join(self.root, gravity_mask_path))
            _gravity_mask_np = np.asarray(gravity_mask)
            if np.sum(_gravity_mask_np == 255) / (_gravity_mask_np.shape[0] * _gravity_mask_np.shape[1]) < 0.02:
                gravity_dir = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)
                w, h = gravity_mask.size
                gravity_mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
        #####################################################################################

        if self._transforms is not None:
            if self.use_gravity:
                img, target, normal_img, mask_img, gravity_mask = self._transforms(img, target, normal_img, mask_img, gravity_mask)
                return img, target, normal_img, mask_img, idx, gravity_dir, gravity_mask
            else:
                img, target, normal_img, mask_img = self._transforms(img, target, normal_img, mask_img)
                return img, target, normal_img, mask_img, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
