from pytorch_local.maskrcnn_benchmark.data import datasets

from .coco import coco_evaluation
from .voc import voc_evaluation

## TODO: Move this to a proper place
# Adding modules for normal evaluation
import torch
import numpy as np
from pytorch_local.maskrcnn_benchmark.data.datasets import COCODataset
import logging
from PIL import Image
import cv2
import os
import os.path


def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))

def evaluate_normal(dataset, predictions, output_folder):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
    Returns:
        evaluation result for each threshold of 11.25, 22.5, 30 for surface normal prediction
    """
    assert isinstance(dataset, COCODataset)
    coco_results = []
    # device = torch.device(cfg.MODEL.DEVICE)
    cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-06)
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("Start evaluating")
    ## TODO: Get device from the prediction tensor
    ## TODO: The image size needs to be a variable rather than 960x1280
    device = torch.device("cuda")
    normal_stats = {'5':{}, '7.5':{}, '10': {}, '11.25':{}, '22.5':{}, '30':{}}
    num_valid_pixels = {'5':{}, '7.5':{}, '10': {}, '11.25':{}, '22.5':{}, '30':{}}
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        ### PREVIOUS CODE TO COMPUTE ERROR OF NORMAL ###
        # normal_img = dataset.get_normal_image(image_id)
        # _gt = normal_img.permute(1, 2, 0).contiguous().view(-1, 3)
        # valid_mask_image_float = _gt[:,0] > 1e-3
        # valid_mask_image_float = valid_mask_image_float.contiguous().view(-1).clone().detach().to(dtype=torch.uint8)
        # if float(torch.sum(valid_mask_image_float, dim=0)) < 1.:
        #      continue
        # tensor_error = cosine_similarity(prediction, _gt)

        ### NEW CODE: #####
        tensor_error = prediction
        valid_mask_image_float = torch.ones_like(tensor_error, dtype=torch.uint8)
        if tensor_error.numel() < 1:
              continue
        tensor_logic_05 = tensor_error.gt(np.cos(5 * np.pi / 180.0))
        tensor_logic_05.mul_(valid_mask_image_float)
        tensor_logic_075 = tensor_error.gt(np.cos(7.5 * np.pi / 180.0))
        tensor_logic_075.mul_(valid_mask_image_float)
        tensor_logic_10 = tensor_error.gt(np.cos(10 * np.pi / 180.0))
        tensor_logic_10.mul_(valid_mask_image_float)
        tensor_logic_1125 = tensor_error.gt(np.cos(11.25 * np.pi/180.0))
        tensor_logic_1125.mul_(valid_mask_image_float)
        tensor_logic_225 = tensor_error.gt(np.cos(22.5 * np.pi / 180.0))
        tensor_logic_225.mul_(valid_mask_image_float)
        tensor_logic_30 = tensor_error.gt(np.cos(30 * np.pi / 180.0))
        tensor_logic_30.mul_(valid_mask_image_float)

        normal_stats['5'].update(
            {original_id: (
                        float(torch.sum(tensor_logic_05, dim=0)) / float(torch.sum(valid_mask_image_float, dim=0)))})
        normal_stats['7.5'].update(
            {original_id: (
                        float(torch.sum(tensor_logic_075, dim=0)) / float(torch.sum(valid_mask_image_float, dim=0)))})
        normal_stats['10'].update(
            {original_id: (
                        float(torch.sum(tensor_logic_10, dim=0)) / float(torch.sum(valid_mask_image_float, dim=0)))})
        normal_stats['11.25'].update(
            {original_id: (float(torch.sum(tensor_logic_1125, dim=0)) / float(torch.sum(valid_mask_image_float, dim=0)))})
        normal_stats['22.5'].update(
            {original_id: (float(torch.sum(tensor_logic_225, dim=0)) / float(torch.sum(valid_mask_image_float, dim=0)))})
        normal_stats['30'].update(
            {original_id: (float(torch.sum(tensor_logic_30, dim=0)) / float(torch.sum(valid_mask_image_float, dim=0)))})

        num_valid_pixels['5'].update({original_id: tensor_error.shape[0]})
        num_valid_pixels['7.5'].update({original_id: tensor_error.shape[0]})
        num_valid_pixels['10'].update({original_id: tensor_error.shape[0]})
        num_valid_pixels['11.25'].update({original_id:tensor_error.shape[0]})
        num_valid_pixels['22.5'].update({original_id: tensor_error.shape[0]})
        num_valid_pixels['30'].update({original_id: tensor_error.shape[0]})

        ## Write binary output image with logic 1 indicating the angle error < 11.25 degree
        # cv2.imwrite(output_folder + 'mask_pred_5deg-%d.png' % (original_id), 255*np.asarray(tensor_logic_1125.reshape(480, 640), dtype=np.uint8))

    import json
    with open('./normal_stat.json', 'w') as fp:
         json.dump(normal_stats, fp)
    logger.info('5: %2.2f' % (100. * np.average(np.asarray(list(normal_stats['5'].values())), axis=0,
                                            weights=np.asarray(list(num_valid_pixels['5'].values())))))
    logger.info('7.5: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['7.5'].values())), axis=0,
                                            weights=np.asarray(list(num_valid_pixels['7.5'].values())))))
    logger.info('10: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['10'].values())), axis=0,
                                            weights=np.asarray(list(num_valid_pixels['10'].values())))))
    logger.info('11.25: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['11.25'].values())), axis=0,
                                            weights=np.asarray(list(num_valid_pixels['11.25'].values())))))
    logger.info('22.5: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['22.5'].values())), axis=0,
                                            weights=np.asarray(list(num_valid_pixels['22.5'].values())))))
    logger.info('30: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['30'].values())), axis=0,
                                            weights=np.asarray(list(num_valid_pixels['30'].values())))))
    return []
