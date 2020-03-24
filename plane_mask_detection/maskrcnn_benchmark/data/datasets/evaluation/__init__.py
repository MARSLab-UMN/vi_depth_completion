from pytorch_local.maskrcnn_benchmark.data import datasets
from pytorch_local.maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
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
import math


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

def _evaluate_normal(dataset, error_normal, normal_prediction, output_folder):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        error_normal(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
    Returns:
        evaluation result for each threshold of 11.25, 22.5, 30 for surface normal prediction
    """
    coco_results = []
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("Start evaluating")

    all_predictions = torch.cat(error_normal)
    all_predictions = torch.clamp(all_predictions, min=-1.0, max=1.0)
    angular_error_predictions = torch.acos(all_predictions) * 180 / np.pi # in degree
    total_test_pixels = torch.tensor(all_predictions.numel(), dtype=torch.float)
    stats = np.asarray([torch.mean(angular_error_predictions),
                        torch.median(angular_error_predictions),
                        torch.sqrt(torch.mean(angular_error_predictions * angular_error_predictions)),
                        ((100.0 * torch.sum(angular_error_predictions.le(5.0))) / total_test_pixels),
                        ((100.0 * torch.sum(angular_error_predictions.le(7.5))) / total_test_pixels),
                        ((100.0 * torch.sum(angular_error_predictions.le(11.25))) / total_test_pixels),
                        ((100.0 * torch.sum(angular_error_predictions.le(22.5))) / total_test_pixels),
                        ((100.0 * torch.sum(angular_error_predictions.le(30.0))) / total_test_pixels)])
    logger.info('mean: %2.4f' % stats[0])
    logger.info('median: %2.4f' % stats[1])
    logger.info('rmse: %2.4f' % stats[2])
    logger.info('5 deg: %2.4f' % stats[3])
    logger.info('7.5 deg: %2.4f' % stats[4])
    logger.info('11.25 deg: %2.4f' % stats[5])
    logger.info('22.55 deg: %2.4f' % stats[6])
    logger.info('30 deg: %2.4f' % stats[7])
    np.savetxt(os.path.join(output_folder, 'stats.txt'), stats, fmt='%.4f', delimiter=' ')

    if normal_prediction is not None:
        logger.info("Start saving images to files")
        for image_id, n in enumerate(normal_prediction):
            original_id = dataset.id_to_img_map[image_id]
            X_dim = 240
            Y_dim = 320
            n_pred = n.reshape(256, Y_dim, 3)
            cv2.imwrite(os.path.join(output_folder, '../%d-pred.jpg' % original_id),
                        np.asarray(127.5 + 127.5 * (n_pred[0:X_dim,:,:]), dtype=np.uint8))

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
    normal_stats = {'5':{}, '7.5':{}, '10': {}, '11.25':{}, '22.5':{}, '30':{}, 'mean':{}, 'median':{}, 'rmse':{}}
    num_valid_pixels = {'5':{}, '7.5':{}, '10': {}, '11.25':{}, '22.5':{}, '30':{}, 'mean':{}, 'median':{}, 'rmse':{}}

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

        valid_mask_image_count = float(torch.sum(valid_mask_image_float, dim=0))

        normal_stats['5'].update(
            {original_id: (float(torch.sum(tensor_logic_05, dim=0)) / valid_mask_image_count)})

        normal_stats['7.5'].update(
            {original_id: (float(torch.sum(tensor_logic_075, dim=0)) / valid_mask_image_count)})

        normal_stats['10'].update(
            {original_id: (float(torch.sum(tensor_logic_10, dim=0)) / valid_mask_image_count)})

        normal_stats['11.25'].update(
            {original_id: (float(torch.sum(tensor_logic_1125, dim=0)) / valid_mask_image_count)})

        normal_stats['22.5'].update(
            {original_id: (float(torch.sum(tensor_logic_225, dim=0)) / valid_mask_image_count)})

        normal_stats['30'].update(
            {original_id: (float(torch.sum(tensor_logic_30, dim=0)) / valid_mask_image_count)})

        #normal_stats['mean'].update(
        #    {original_id: (float(torch.sum(tensor_error, dim=0)) / valid_mask_image_count)})

        #normal_stats['median'].update(
        #    {original_id: (float(torch.median(tensor_error)))})

        #normal_stats['rmse'].update(
        #    {original_id: (math.sqrt(float(torch.sum(tensor_error * tensor_error, dim=0)) / valid_mask_image_count))})


        num_valid_pixels['5'].update({original_id: tensor_error.shape[0]})
        num_valid_pixels['7.5'].update({original_id: tensor_error.shape[0]})
        num_valid_pixels['10'].update({original_id: tensor_error.shape[0]})
        num_valid_pixels['11.25'].update({original_id:tensor_error.shape[0]})
        num_valid_pixels['22.5'].update({original_id: tensor_error.shape[0]})
        num_valid_pixels['30'].update({original_id: tensor_error.shape[0]})
        #num_valid_pixels['mean'].update({original_id: tensor_error.shape[0]})
        #num_valid_pixels['median'].update({original_id: tensor_error.shape[0]})
        #num_valid_pixels['rmse'].update({original_id: tensor_error.shape[0]})
        ## Write binary output image with logic 1 indicating the angle error < 11.25 degree
        # cv2.imwrite(output_folder + 'mask_pred_5deg-%d.png' % (original_id), 255*np.asarray(tensor_logic_1125.reshape(480, 640), dtype=np.uint8))

    #for i in range(num_tensor):
    #    if tensor_error_list[i].shape[0] > 1:
    #        tensor_error_list[i] = tensor_error_list[i][1:]
    #    else:
    #        del tensor_error_list[i]
    #    print (tensor_error_list[i].shape)
    import json
    with open('./normal_stat.json', 'w') as fp:
         json.dump(normal_stats, fp)
    weights_pixels = np.asarray(list(num_valid_pixels['5'].values()))


    logger.info('5: %2.2f' % (100. * np.average(np.asarray(list(normal_stats['5'].values())), axis=0,
                                            weights=weights_pixels)))
    logger.info('7.5: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['7.5'].values())), axis=0,
                                            weights=weights_pixels)))
    logger.info('10: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['10'].values())), axis=0,
                                            weights=weights_pixels)))
    logger.info('11.25: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['11.25'].values())), axis=0,
                                            weights=weights_pixels)))
    logger.info('22.5: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['22.5'].values())), axis=0,
                                            weights=weights_pixels)))
    logger.info('30: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['30'].values())), axis=0,
                                            weights=weights_pixels)))
    #logger.info('mean: %2.2f' % (180.* np.arccos(np.average(np.asarray(list(normal_stats['mean'].values())), axis=0,
    #                                        weights=weights_pixels)) / np.pi))
    #logger.info('median: %2.2f' % (180.* np.arccos(np.median(np.asarray(list(normal_stats['median'].values())), axis=0)) / np.pi))
    #logger.info('rmse: %2.2f' % (180.* np.arccos(np.average(np.asarray(list(normal_stats['rmse'].values())), axis=0,
    #                                        weights=weights_pixels)) / np.pi))

    return []

def evaluate_plane_normal_consistency(dataset, normal_predictions, output_folder):

    assert isinstance(dataset, COCODataset)
    coco_results = []
    # device = torch.device(cfg.MODEL.DEVICE)
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("Start evaluating plane-normal consistency")
    device = torch.device("cuda")
    normal_stats = {'5':{}, '7.5':{}, '10': {}, '11.25':{}, '22.5':{}, '30':{}}
    num_valid_pixels = {'5':{}, '7.5':{}, '10': {}, '11.25':{}, '22.5':{}, '30':{}}

    # normal_predictions = normal_predictions[:1]
    for image_id, normal_pred in enumerate(normal_predictions):
        if image_id != 16 and image_id != 1210 and image_id != 2015 and image_id != 2020:
            continue
        original_id = dataset.id_to_img_map[image_id]
        img_info = dataset.get_img_info(image_id)
        print (img_info)
        height = img_info["width"]
        width = img_info["height"]
        normal_gt, mask_gt = dataset.get_normal_image(image_id)
        if len(np.unique(mask_gt)) == 1:
            continue
        # Getting the normal_pred
        output_normal_prediction = normal_pred.reshape(480, 640, 3)
        normalized_normal_image_pred = torch.nn.functional.normalize(output_normal_prediction, dim=2)
        normalized_normal_image_pred = normalized_normal_image_pred.double()

        tt_pred = torch.atan2(normalized_normal_image_pred[:, :, 1], normalized_normal_image_pred[:, :, 0])
        phi_pred = torch.atan2(normalized_normal_image_pred[:, :, 2],
                               torch.sqrt(torch.mul(normalized_normal_image_pred[:, :, 0], normalized_normal_image_pred[:, :, 0])
                                          + torch.mul(normalized_normal_image_pred[:, :, 1],
                                                      normalized_normal_image_pred[:, :, 1])))

        # Getting the normal gt
        normal_gt = normal_gt.permute(1, 2, 0).numpy()
        output_normal_gt = torch.from_numpy(normal_gt)
        normalized_normal_image_gt = torch.nn.functional.normalize(output_normal_gt, dim=2)
        normalized_normal_image_gt = normalized_normal_image_gt.double()

        tt_gt = torch.atan2(normalized_normal_image_gt[:, :, 1], normalized_normal_image_gt[:, :, 0])
        phi_gt = torch.atan2(normalized_normal_image_gt[:, :, 2],
                               torch.sqrt(torch.mul(normalized_normal_image_gt[:, :, 0], normalized_normal_image_gt[:, :, 0])
                                          + torch.mul(normalized_normal_image_gt[:, :, 1],
                                                      normalized_normal_image_gt[:, :, 1])))

        for index in range(1, len(np.unique(mask_gt))):
            index_mask = (mask_gt == index).astype(np.uint8)
            mean_tt_over_mask = torch.sum(tt_gt[index_mask])
            mean_tt_over_mask /= tt_gt[index_mask].shape[0]
            mean_phi_over_mask = torch.sum(phi_gt[index_mask])
            mean_phi_over_mask /= phi_gt[index_mask].shape[0]

            n_mean_gt = torch.zeros([3, 1], dtype=torch.double)
            n_mean_gt[0] = torch.cos(mean_phi_over_mask) * torch.cos(mean_tt_over_mask)
            n_mean_gt[1] = torch.cos(mean_phi_over_mask) * torch.sin(mean_tt_over_mask)
            n_mean_gt[2] = torch.sin(mean_phi_over_mask)
            n_mean_gt = torch.nn.functional.normalize(n_mean_gt, dim=1)

            gt_error_over_mask = torch.mm(normalized_normal_image_gt[index_mask], n_mean_gt)
            tensor_logic_05_gt = gt_error_over_mask.gt(np.cos(5 * np.pi / 180.0))
            print(gt_error_over_mask.shape)
            percent_lt_05_gt = 100. * float(torch.sum(tensor_logic_05_gt, dim=0)) / index_mask.sum()
            print('gt 5: %2.2f' % percent_lt_05_gt)

            if index_mask.sum() > 60000 and percent_lt_05_gt > 80:
                print('the gt_normal inside this gt_mask is very good')

                pred_mean_tt_over_mask = torch.sum(tt_pred[index_mask])
                pred_mean_tt_over_mask /= tt_pred[index_mask].shape[0]
                pred_mean_phi_over_mask = torch.sum(phi_pred[index_mask])
                pred_mean_phi_over_mask /= phi_pred[index_mask].shape[0]

                n_mean_pred = torch.zeros([3, 1], dtype=torch.double)
                n_mean_pred[0] = torch.cos(pred_mean_phi_over_mask) * torch.cos(pred_mean_tt_over_mask)
                n_mean_pred[1] = torch.cos(pred_mean_phi_over_mask) * torch.sin(pred_mean_tt_over_mask)
                n_mean_pred[2] = torch.sin(pred_mean_phi_over_mask)
                n_mean_pred = torch.nn.functional.normalize(n_mean_pred, dim=1)

                pred_error_over_mask = torch.mm(normalized_normal_image_pred[index_mask], n_mean_pred)
                tensor_logic_05_pred = pred_error_over_mask.gt(np.cos(5 * np.pi / 180.0))
                print(pred_error_over_mask.shape)
                percent_lt_05_pred = 100. * float(torch.sum(tensor_logic_05_pred, dim=0)) / index_mask.sum()
                print('pred 5: %2.2f' % percent_lt_05_pred)

                if percent_lt_05_pred <= 100:
                    print('the pred_normal inside this gt_mask is bad, save this')

                    if not cv2.imwrite('plane_normal_const/%06d-normal_gt_%s.jpg' % (image_id, str(original_id)),
                                       np.asarray(127.5 + 127.5 * normalized_normal_image_gt.numpy(), dtype=np.uint8)):
                        print('saving normal_gt failed')

                    if not cv2.imwrite('plane_normal_const/%06d-normal_pred_%s.jpg' % (image_id, str(original_id)),
                                       np.asarray(127.5 + 127.5 * normalized_normal_image_pred.numpy(), dtype=np.uint8)):
                        print('saving normal_pred failed')

                    mask_gt_img = np.copy(127.5 + 127.5 * normalized_normal_image_gt.numpy())
                    mask_gt_img = np.asarray(mask_gt_img, dtype=float)
                    for planeIdx in range(1, len(np.unique(mask_gt))):
                        mask_gt_img[mask_gt == planeIdx] *= 0.5
                        mask_gt_img[mask_gt == planeIdx] += 0.5 * np.array([np.random.randint(low=0, high=255),
                                                                            np.random.randint(low=0, high=255),
                                                                            np.random.randint(low=0, high=255)])

                    if not cv2.imwrite('plane_normal_const/%06d-mask_gt_%s.jpg' % (image_id, str(original_id)),
                                       mask_gt_img):
                        print('saving mask_gt failed')

                    mask_gt_img = np.zeros_like(normalized_normal_image_gt.numpy())

                    mask_gt_img[mask_gt == index] = np.array([np.random.randint(low=0, high=255),
                                                              np.random.randint(low=0, high=255),
                                                              np.random.randint(low=0, high=255)])
                    if not cv2.imwrite('plane_normal_const/%06d-mask_%s_5deg.jpg' % (image_id, str(original_id)),
                                       mask_gt_img):
                        print('saving individual mask_gt failed')

        # if not cv2.imwrite('plane_normal_const/%06d-normal_gt.jpg' % image_id,
        #             np.asarray(127.5 + 127.5 * normal_gt, dtype=np.uint8)):
        #     print ('saving failed')
        # mask_gt_img = np.copy(127.5 + 127.5 * normal_gt)
        # mask_gt_img = np.asarray(mask_gt_img, dtype=float)
        # print (mask_gt.shape)
        # print (mask_gt_img.shape)
        # for index in range(1, len(np.unique(mask_gt))):
        #     mask_gt_img[mask_gt == index] *= 0.5
        #     mask_gt_img[mask_gt == index] += 0.5 * np.array([np.random.randint(low=0, high=255),
        #                                                      np.random.randint(low=0, high=255),
        #                                                      np.random.randint(low=0, high=255)])
        #
        # if not cv2.imwrite('plane_normal_const/%06d-mask_gt.jpg' % image_id, mask_gt_img):
        #     print ('saving failed')

        # output_normal_prediction = normal_pred.numpy().reshape(480, 640, 3)
        # if not cv2.imwrite('plane_normal_const/%06d-normal_pred.jpg' % image_id,
        #                    np.asarray(127.5 + 127.5 * output_normal_prediction, dtype=np.uint8)):
        #     print ('saving failed')

        # output_normal_gt = torch.from_numpy(normal_gt)
        # normalized_normal_image = torch.nn.functional.normalize(output_normal_gt, dim=2)
        # normalized_normal_image = normalized_normal_image.double()
        #
        # tt_pred = torch.atan2(normalized_normal_image[:, :, 1], normalized_normal_image[:, :, 0])
        # phi_pred = torch.atan2(normalized_normal_image[:, :, 2],
        #                        torch.sqrt(torch.mul(normalized_normal_image[:, :, 0], normalized_normal_image[:, :, 0])
        #                                   + torch.mul(normalized_normal_image[:, :, 1],
        #                                               normalized_normal_image[:, :, 1])))
        # n_mean = torch.zeros_like(normalized_normal_image)
        #
        # for index in range(1, len(np.unique(mask_gt))):
        #     index_mask = (mask_gt == index).astype(np.uint8)
        #     mean_tt_over_mask = torch.sum(tt_pred[index_mask])
        #     mean_tt_over_mask /= tt_pred[index_mask].shape[0]
        #     mean_phi_over_mask = torch.sum(phi_pred[index_mask])
        #     mean_phi_over_mask /= phi_pred[index_mask].shape[0]
        #     n_mean =  torch.zeros([3, 1], dtype=torch.double)
        #     n_mean[0] = torch.cos(mean_phi_over_mask) * torch.cos(mean_tt_over_mask)
        #     n_mean[1] = torch.cos(mean_phi_over_mask) * torch.sin(mean_tt_over_mask)
        #     n_mean[2] = torch.sin(mean_phi_over_mask)
        #     n_mean = torch.nn.functional.normalize(n_mean, dim=1)
        #     error_over_mask = torch.mm(normalized_normal_image[index_mask], n_mean)
        #     tensor_logic_05 = error_over_mask.gt(np.cos(5 * np.pi / 180.0))
        #     print (error_over_mask.shape)
        #     percent_lt_05 = 100. * float(torch.sum(tensor_logic_05, dim=0)) / index_mask.sum()
        #     print('5: %2.2f' % percent_lt_05)
        #
        #     if index_mask.sum() > 20000 and percent_lt_05 < 80:
        #         print ('this mask is not good')
        #         # mask_gt_img = np.copy(127.5 + 127.5 * normal_gt)
        #         if not cv2.imwrite('plane_normal_const/%06d-normal_gt_%s.jpg' % (image_id, str(original_id)),
        #                            np.asarray(127.5 + 127.5 * normal_gt, dtype=np.uint8)):
        #             print('saving failed')
        #         mask_gt_img = np.copy(127.5 + 127.5 * normal_gt)
        #         mask_gt_img = np.asarray(mask_gt_img, dtype=float)
        #         print(mask_gt.shape)
        #         print(mask_gt_img.shape)
        #         for planeIdx in range(1, len(np.unique(mask_gt))):
        #             mask_gt_img[mask_gt == planeIdx] *= 0.5
        #             mask_gt_img[mask_gt == planeIdx] += 0.5 * np.array([np.random.randint(low=0, high=255),
        #                                                              np.random.randint(low=0, high=255),
        #                                                              np.random.randint(low=0, high=255)])
        #
        #         if not cv2.imwrite('plane_normal_const/%06d-mask_gt.jpg' % image_id, mask_gt_img):
        #             print('saving failed')
        #
        #         mask_gt_img = np.zeros_like(normal_gt)
        #
        #         mask_gt_img[mask_gt == index] = np.array([np.random.randint(low=0, high=255),
        #                                                   np.random.randint(low=0, high=255),
        #                                                   np.random.randint(low=0, high=255)])
        #         if not cv2.imwrite('plane_normal_const/%06d-mask_%01d_5deg.jpg' % (image_id, index), mask_gt_img):
        #             print('saving failed')

        # # print (tt_pred.shape)
        # # print (phi_pred.shape)
        # # print (mask_gt.shape)
        # var_tt_tensor = torch.zeros_like(tt_pred)
        # var_phi_tensor = torch.zeros_like(phi_pred)
        # print ('Number of masks: ', len(np.unique(mask_gt)))
        # num_pixels = 0
        # valid_mask = np.zeros_like(mask_gt, dtype=np.uint8)
        # for index in range(1, len(np.unique(mask_gt))):
        #     index_mask = (mask_gt == index).astype(np.uint8)
        #     mean_tt_over_mask = torch.sum(tt_pred[index_mask])
        #     mean_tt_over_mask /= tt_pred[index_mask].shape[0]
        #     var_tt_tensor[index_mask] = (tt_pred[index_mask] - mean_tt_over_mask).abs()
        #     var_tt_over_mask = (tt_pred[index_mask] - mean_tt_over_mask).abs()
        #     var_tt_over_mask /= tt_pred[index_mask].shape[0]
        #
        #     mean_phi_over_mask = torch.sum(phi_pred[index_mask])
        #     mean_phi_over_mask /= phi_pred[index_mask].shape[0]
        #     var_phi_tensor[index_mask] = (phi_pred[index_mask] - mean_phi_over_mask).abs()
        #     var_phi_over_mask = (phi_pred[index_mask] - mean_phi_over_mask).abs()
        #     var_phi_over_mask /= phi_pred[index_mask].shape[0]
        #
        #     num_pixels += index_mask.sum()
        #     valid_mask += index_mask
        #
        #     # print('mean_tt/phi over mask: ', mean_tt_over_mask, '/', mean_phi_over_mask)
        #     print('var_tt/phi over mask: ', torch.sum(var_tt_over_mask), '/', torch.sum(var_phi_over_mask))
        #
        #     tensor_tt_error = var_tt_tensor[index_mask]
        #     if tensor_tt_error.numel() < 1:
        #         continue
        #
        #     tensor_logic_05 = tensor_tt_error.lt(30 * np.pi / 180.0)
        #     percent_tt_lt_05 = 100.* float(torch.sum(tensor_logic_05, dim=0)) / tt_pred[index_mask].shape[0]
        #     print ('tt 5: %2.2f' % percent_tt_lt_05)
        #
        #     tensor_phi_error = var_phi_tensor[index_mask]
        #     if tensor_phi_error.numel() < 1:
        #         continue
        #
        #     tensor_logic_05 = tensor_phi_error.lt(30 * np.pi / 180.0)
        #     percent_phi_lt_05 = 100. * float(torch.sum(tensor_logic_05, dim=0)) / phi_pred[index_mask].shape[0]
        #     print('phi 5: %2.2f' % percent_phi_lt_05)
        #
        #     if tt_pred[index_mask].shape[0] > 20000 and (percent_tt_lt_05 < 40 or percent_phi_lt_05 < 40):
        #         print ('this mask is not good')
        #         # mask_gt_img = np.copy(127.5 + 127.5 * normal_gt)
        #         if not cv2.imwrite('plane_normal_const/%06d-normal_gt_%s.jpg' % (image_id, str(original_id)),
        #                            np.asarray(127.5 + 127.5 * normal_gt, dtype=np.uint8)):
        #             print('saving failed')
        #         mask_gt_img = np.copy(127.5 + 127.5 * normal_gt)
        #         mask_gt_img = np.asarray(mask_gt_img, dtype=float)
        #         print(mask_gt.shape)
        #         print(mask_gt_img.shape)
        #         for planeIdx in range(1, len(np.unique(mask_gt))):
        #             mask_gt_img[mask_gt == planeIdx] *= 0.5
        #             mask_gt_img[mask_gt == planeIdx] += 0.5 * np.array([np.random.randint(low=0, high=255),
        #                                                              np.random.randint(low=0, high=255),
        #                                                              np.random.randint(low=0, high=255)])
        #
        #         if not cv2.imwrite('plane_normal_const/%06d-mask_gt.jpg' % image_id, mask_gt_img):
        #             print('saving failed')
        #
        #         mask_gt_img = np.zeros_like(normal_gt)
        #
        #         mask_gt_img[mask_gt == index] = np.array([np.random.randint(low=0, high=255),
        #                                                   np.random.randint(low=0, high=255),
        #                                                   np.random.randint(low=0, high=255)])
        #         if not cv2.imwrite('plane_normal_const/%06d-mask_%01d_5deg.jpg' % (image_id, index), mask_gt_img):
        #             print('saving failed')
        #

########################################### comment
        # tensor_error = var_tt_tensor[valid_mask]
        #
        # valid_mask_image_float = torch.ones_like(tensor_error, dtype=torch.uint8)
        #
        # if tensor_error.numel() < 1:
        #       continue
        # tensor_logic_05 = tensor_error.lt(5 * np.pi / 180.0)
        # tensor_logic_05.mul_(valid_mask_image_float)
        # tensor_logic_075 = tensor_error.lt(7.5 * np.pi / 180.0)
        # tensor_logic_075.mul_(valid_mask_image_float)
        # tensor_logic_10 = tensor_error.lt(10 * np.pi / 180.0)
        # tensor_logic_10.mul_(valid_mask_image_float)
        # tensor_logic_1125 = tensor_error.lt(11.25 * np.pi/180.0)
        # tensor_logic_1125.mul_(valid_mask_image_float)
        # tensor_logic_225 = tensor_error.lt(22.5 * np.pi / 180.0)
        # tensor_logic_225.mul_(valid_mask_image_float)
        # tensor_logic_30 = tensor_error.lt(30 * np.pi / 180.0)
        # tensor_logic_30.mul_(valid_mask_image_float)
        #
        # valid_mask_image_count = float(torch.sum(valid_mask_image_float, dim=0))
        #
        # print ('5: %2.2f', (float(torch.sum(tensor_logic_05, dim=0)) / valid_mask_image_count))

        # normalized_normal_image = torch.nn.functional.normalize(output_normal_prediction, dim=2)
        # normalized_normal_image = normalized_normal_image.double()
        #
        # tt_pred = torch.atan2(normalized_normal_image[:, :, 1], normalized_normal_image[:, :, 0])
        # phi_pred = torch.atan2(normalized_normal_image[:, :, 2],
        #                        torch.sqrt(torch.mul(normalized_normal_image[:, :, 0], normalized_normal_image[:, :, 0])
        #                                   + torch.mul(normalized_normal_image[:, :, 1], normalized_normal_image[:, :, 1])))

        # for mask in masks:
        #     index_mask = mask
        #     mean_tt_over_mask = torch.sum(tt_pred[index_mask])
        #     mean_tt_over_mask /= tt_pred[index_mask].shape[0]
        #     var_tt_tensor[index_mask] = (tt_pred[index_mask] - mean_tt_over_mask).abs()
        #     var_tt_over_mask = (tt_pred[index_mask] - mean_tt_over_mask).abs()
        #     var_tt_over_mask /= tt_pred[index_mask].shape[0]
        #
        #     mean_phi_over_mask = torch.sum(phi_pred[index_mask])
        #     mean_phi_over_mask /= phi_pred[index_mask].shape[0]
        #     var_phi_tensor[index_mask] = (phi_pred[index_mask] - mean_phi_over_mask).abs()
        #     var_phi_over_mask = (phi_pred[index_mask] - mean_phi_over_mask).abs()
        #     var_phi_over_mask /= phi_pred[index_mask].shape[0]
        #
        #     num_pixels += index_mask.sum()
        #     valid_mask += index_mask

            #print('mean_tt/phi over mask: ', mean_tt_over_mask, '/', mean_phi_over_mask)
            #print('var_tt/phi over mask: ', torch.sum(var_tt_over_mask), '/', torch.sum(var_phi_over_mask))

        #normal_gt = normal_gt.permute(1, 2, 0).contiguous().view(-1, 3)
    #     mask_pred = predictions[image_id]
    #     mask_pred = mask_pred.resize((height, width))
    #
    #     if mask_pred.has_field("mask"):
    #         # if we have masks, paste the masks in the right position
    #         # in the image, as defined by the bounding boxes
    #         masks = mask_pred.get_field("mask")
    #         # always single image is passed at a time
    #         masker = Masker(threshold=0.5, padding=1)
    #         masks = masker([masks], [mask_pred])[0]
    #         mask_pred.add_field("mask", masks)
    #
    #     # Getting the normal in the image form
    #     output_normal_prediction = normal_pred.reshape(480, 640, 3)
    #     # mask_pred (predictions), normal_pred, output_normal_prediction
    #     top_predictions = select_top_predictions(mask_pred)
    #     masks = top_predictions.get_field("mask").numpy()
    #     normalized_normal_image = torch.nn.functional.normalize(output_normal_prediction, dim=2)
    #     normalized_normal_image = normalized_normal_image.double()
    #
    #     tt_pred = torch.atan2(normalized_normal_image[:, :, 1], normalized_normal_image[:, :, 0])
    #     phi_pred = torch.atan2(normalized_normal_image[:, :, 2],
    #                        torch.sqrt(torch.mul(normalized_normal_image[:, :, 0], normalized_normal_image[:, :, 0])
    #                                   + torch.mul(normalized_normal_image[:, :, 1], normalized_normal_image[:, :, 1])))
    #
    #     var_tt_tensor = torch.zeros_like(tt_pred)
    #     var_phi_tensor = torch.zeros_like(phi_pred)
    #     num_pixels = 0
    #     valid_mask = np.zeros((1, 480, 640), dtype=np.uint8)
    #     for mask in masks:
    #         index_mask = mask
    #         mean_tt_over_mask = torch.sum(tt_pred[index_mask])
    #         mean_tt_over_mask /= tt_pred[index_mask].shape[0]
    #         var_tt_tensor[index_mask] = (tt_pred[index_mask] - mean_tt_over_mask).abs()
    #         var_tt_over_mask = (tt_pred[index_mask] - mean_tt_over_mask).abs()
    #         var_tt_over_mask /= tt_pred[index_mask].shape[0]
    #
    #         mean_phi_over_mask = torch.sum(phi_pred[index_mask])
    #         mean_phi_over_mask /= phi_pred[index_mask].shape[0]
    #         var_phi_tensor[index_mask] = (phi_pred[index_mask] - mean_phi_over_mask).abs()
    #         var_phi_over_mask = (phi_pred[index_mask] - mean_phi_over_mask).abs()
    #         var_phi_over_mask /= phi_pred[index_mask].shape[0]
    #
    #         num_pixels += index_mask.sum()
    #         valid_mask += index_mask
    #
    #         #print('mean_tt/phi over mask: ', mean_tt_over_mask, '/', mean_phi_over_mask)
    #         #print('var_tt/phi over mask: ', torch.sum(var_tt_over_mask), '/', torch.sum(var_phi_over_mask))
    #
    #     #for mask in masks:
    #     #    index_mask = mask
    #     #    print ('var_tt/phi over mask: ', torch.sum(var_tt_tensor[index_mask])/index_mask.sum(), '/', torch.sum(var_phi_tensor[index_mask])/index_mask.sum())
    #
    #     tensor_error = var_tt_tensor[valid_mask]
    #     #print ('var_tt/phi for image: ', torch.sum(var_tt_tensor)/num_pixels, '/', torch.sum(var_phi_tensor)/num_pixels)
    #
    #     valid_mask_image_float = torch.ones_like(tensor_error, dtype=torch.uint8)
    #
    #     if tensor_error.numel() < 1:
    #           continue
    #     tensor_logic_05 = tensor_error.lt(5 * np.pi / 180.0)
    #     tensor_logic_05.mul_(valid_mask_image_float)
    #     tensor_logic_075 = tensor_error.lt(7.5 * np.pi / 180.0)
    #     tensor_logic_075.mul_(valid_mask_image_float)
    #     tensor_logic_10 = tensor_error.lt(10 * np.pi / 180.0)
    #     tensor_logic_10.mul_(valid_mask_image_float)
    #     tensor_logic_1125 = tensor_error.lt(11.25 * np.pi/180.0)
    #     tensor_logic_1125.mul_(valid_mask_image_float)
    #     tensor_logic_225 = tensor_error.lt(22.5 * np.pi / 180.0)
    #     tensor_logic_225.mul_(valid_mask_image_float)
    #     tensor_logic_30 = tensor_error.lt(30 * np.pi / 180.0)
    #     tensor_logic_30.mul_(valid_mask_image_float)
    #
    #     valid_mask_image_count = float(torch.sum(valid_mask_image_float, dim=0))
    #
    #     normal_stats['5'].update(
    #         {original_id: (float(torch.sum(tensor_logic_05, dim=0)) / valid_mask_image_count)})
    #
    #     normal_stats['7.5'].update(
    #         {original_id: (float(torch.sum(tensor_logic_075, dim=0)) / valid_mask_image_count)})
    #
    #     normal_stats['10'].update(
    #         {original_id: (float(torch.sum(tensor_logic_10, dim=0)) / valid_mask_image_count)})
    #
    #     normal_stats['11.25'].update(
    #         {original_id: (float(torch.sum(tensor_logic_1125, dim=0)) / valid_mask_image_count)})
    #
    #     normal_stats['22.5'].update(
    #         {original_id: (float(torch.sum(tensor_logic_225, dim=0)) / valid_mask_image_count)})
    #
    #     normal_stats['30'].update(
    #         {original_id: (float(torch.sum(tensor_logic_30, dim=0)) / valid_mask_image_count)})
    #
    #     num_valid_pixels['5'].update({original_id: tensor_error.shape[0]})
    #     num_valid_pixels['7.5'].update({original_id: tensor_error.shape[0]})
    #     num_valid_pixels['10'].update({original_id: tensor_error.shape[0]})
    #     num_valid_pixels['11.25'].update({original_id:tensor_error.shape[0]})
    #     num_valid_pixels['22.5'].update({original_id: tensor_error.shape[0]})
    #     num_valid_pixels['30'].update({original_id: tensor_error.shape[0]})
    #
    #
    # weights_pixels = np.asarray(list(num_valid_pixels['5'].values()))
    #
    #
    # logger.info('5: %2.2f' % (100. * np.average(np.asarray(list(normal_stats['5'].values())), axis=0,
    #                                         weights=weights_pixels)))
    # logger.info('7.5: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['7.5'].values())), axis=0,
    #                                         weights=weights_pixels)))
    # logger.info('10: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['10'].values())), axis=0,
    #                                         weights=weights_pixels)))
    # logger.info('11.25: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['11.25'].values())), axis=0,
    #                                         weights=weights_pixels)))
    # logger.info('22.5: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['22.5'].values())), axis=0,
    #                                         weights=weights_pixels)))
    # logger.info('30: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['30'].values())), axis=0,
    #                                         weights=weights_pixels)))

    return []

def _evaluate_plane_normal_consistency(dataset, predictions, normal_predictions, output_folder):

    assert isinstance(dataset, COCODataset)
    coco_results = []
    # device = torch.device(cfg.MODEL.DEVICE)
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("Start evaluating plane-normal consistency")
    device = torch.device("cuda")
    normal_stats = {'5':{}, '7.5':{}, '10': {}, '11.25':{}, '22.5':{}, '30':{}}
    num_valid_pixels = {'5':{}, '7.5':{}, '10': {}, '11.25':{}, '22.5':{}, '30':{}}


    for image_id, normal_pred in enumerate(normal_predictions):
        original_id = dataset.id_to_img_map[image_id]
        img_info = dataset.get_img_info(image_id)
        print (img_info)
        height = img_info["width"]
        width = img_info["height"]
        mask_pred = predictions[image_id]
        mask_pred = mask_pred.resize((height, width))

        if mask_pred.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = mask_pred.get_field("mask")
            # always single image is passed at a time
            masker = Masker(threshold=0.5, padding=1)
            masks = masker([masks], [mask_pred])[0]
            mask_pred.add_field("mask", masks)

        # Getting the normal in the image form
        output_normal_prediction = normal_pred.reshape(480, 640, 3)
        # mask_pred (predictions), normal_pred, output_normal_prediction
        top_predictions = select_top_predictions(mask_pred)
        masks = top_predictions.get_field("mask").numpy()
        normalized_normal_image = torch.nn.functional.normalize(output_normal_prediction, dim=2)
        normalized_normal_image = normalized_normal_image.double()

        tt_pred = torch.atan2(normalized_normal_image[:, :, 1], normalized_normal_image[:, :, 0])
        phi_pred = torch.atan2(normalized_normal_image[:, :, 2],
                           torch.sqrt(torch.mul(normalized_normal_image[:, :, 0], normalized_normal_image[:, :, 0])
                                      + torch.mul(normalized_normal_image[:, :, 1], normalized_normal_image[:, :, 1])))

        var_tt_tensor = torch.zeros_like(tt_pred)
        var_phi_tensor = torch.zeros_like(phi_pred)
        num_pixels = 0
        valid_mask = np.zeros((1, 480, 640), dtype=np.uint8)
        for mask in masks:
            index_mask = mask
            mean_tt_over_mask = torch.sum(tt_pred[index_mask])
            mean_tt_over_mask /= tt_pred[index_mask].shape[0]
            var_tt_tensor[index_mask] = (tt_pred[index_mask] - mean_tt_over_mask).abs()
            var_tt_over_mask = (tt_pred[index_mask] - mean_tt_over_mask).abs()
            var_tt_over_mask /= tt_pred[index_mask].shape[0]

            mean_phi_over_mask = torch.sum(phi_pred[index_mask])
            mean_phi_over_mask /= phi_pred[index_mask].shape[0]
            var_phi_tensor[index_mask] = (phi_pred[index_mask] - mean_phi_over_mask).abs()
            var_phi_over_mask = (phi_pred[index_mask] - mean_phi_over_mask).abs()
            var_phi_over_mask /= phi_pred[index_mask].shape[0]

            num_pixels += index_mask.sum()
            valid_mask += index_mask

            #print('mean_tt/phi over mask: ', mean_tt_over_mask, '/', mean_phi_over_mask)
            #print('var_tt/phi over mask: ', torch.sum(var_tt_over_mask), '/', torch.sum(var_phi_over_mask))

        #for mask in masks:
        #    index_mask = mask
        #    print ('var_tt/phi over mask: ', torch.sum(var_tt_tensor[index_mask])/index_mask.sum(), '/', torch.sum(var_phi_tensor[index_mask])/index_mask.sum())

        tensor_error = var_tt_tensor[valid_mask]
        #print ('var_tt/phi for image: ', torch.sum(var_tt_tensor)/num_pixels, '/', torch.sum(var_phi_tensor)/num_pixels)

        valid_mask_image_float = torch.ones_like(tensor_error, dtype=torch.uint8)

        if tensor_error.numel() < 1:
              continue
        tensor_logic_05 = tensor_error.lt(5 * np.pi / 180.0)
        tensor_logic_05.mul_(valid_mask_image_float)
        tensor_logic_075 = tensor_error.lt(7.5 * np.pi / 180.0)
        tensor_logic_075.mul_(valid_mask_image_float)
        tensor_logic_10 = tensor_error.lt(10 * np.pi / 180.0)
        tensor_logic_10.mul_(valid_mask_image_float)
        tensor_logic_1125 = tensor_error.lt(11.25 * np.pi/180.0)
        tensor_logic_1125.mul_(valid_mask_image_float)
        tensor_logic_225 = tensor_error.lt(22.5 * np.pi / 180.0)
        tensor_logic_225.mul_(valid_mask_image_float)
        tensor_logic_30 = tensor_error.lt(30 * np.pi / 180.0)
        tensor_logic_30.mul_(valid_mask_image_float)

        valid_mask_image_count = float(torch.sum(valid_mask_image_float, dim=0))

        normal_stats['5'].update(
            {original_id: (float(torch.sum(tensor_logic_05, dim=0)) / valid_mask_image_count)})

        normal_stats['7.5'].update(
            {original_id: (float(torch.sum(tensor_logic_075, dim=0)) / valid_mask_image_count)})

        normal_stats['10'].update(
            {original_id: (float(torch.sum(tensor_logic_10, dim=0)) / valid_mask_image_count)})

        normal_stats['11.25'].update(
            {original_id: (float(torch.sum(tensor_logic_1125, dim=0)) / valid_mask_image_count)})

        normal_stats['22.5'].update(
            {original_id: (float(torch.sum(tensor_logic_225, dim=0)) / valid_mask_image_count)})

        normal_stats['30'].update(
            {original_id: (float(torch.sum(tensor_logic_30, dim=0)) / valid_mask_image_count)})

        num_valid_pixels['5'].update({original_id: tensor_error.shape[0]})
        num_valid_pixels['7.5'].update({original_id: tensor_error.shape[0]})
        num_valid_pixels['10'].update({original_id: tensor_error.shape[0]})
        num_valid_pixels['11.25'].update({original_id:tensor_error.shape[0]})
        num_valid_pixels['22.5'].update({original_id: tensor_error.shape[0]})
        num_valid_pixels['30'].update({original_id: tensor_error.shape[0]})


    weights_pixels = np.asarray(list(num_valid_pixels['5'].values()))


    logger.info('5: %2.2f' % (100. * np.average(np.asarray(list(normal_stats['5'].values())), axis=0,
                                            weights=weights_pixels)))
    logger.info('7.5: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['7.5'].values())), axis=0,
                                            weights=weights_pixels)))
    logger.info('10: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['10'].values())), axis=0,
                                            weights=weights_pixels)))
    logger.info('11.25: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['11.25'].values())), axis=0,
                                            weights=weights_pixels)))
    logger.info('22.5: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['22.5'].values())), axis=0,
                                            weights=weights_pixels)))
    logger.info('30: %2.2f' % (100.* np.average(np.asarray(list(normal_stats['30'].values())), axis=0,
                                            weights=weights_pixels)))

    return []

def evaluate_plane_normal(dataset, predictions, normal_predictions, output_folder):
    assert isinstance(dataset, COCODataset)
    coco_results = []
    # device = torch.device(cfg.MODEL.DEVICE)
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("Start evaluating plane-normal consistency")
    device = torch.device("cuda")
    normal_stats = {'5':{}, '7.5':{}, '10': {}, '11.25':{}, '22.5':{}, '30':{}}
    num_valid_pixels = {'5':{}, '7.5':{}, '10': {}, '11.25':{}, '22.5':{}, '30':{}}

    for image_id, normal_pred in enumerate(normal_predictions):
        original_id = dataset.id_to_img_map[image_id]
        img_info = dataset.get_img_info(image_id)
        print (img_info)
        height = img_info["width"]
        width = img_info["height"]
        mask_pred = predictions[image_id]
        # print ('mask_pred:', mask_pred)
        mask_pred = mask_pred.resize((height, width))

        if mask_pred.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = mask_pred.get_field("mask")
            # always single image is passed at a time
            masker = Masker(threshold=0.5, padding=1)
            masks = masker([masks], [mask_pred])[0]
            mask_pred.add_field("mask", masks)
        print ('mask_pred:', mask_pred)

        path = '/mars/mnt/dgx/FrameNet/scannet-frames/' + img_info['file_name']
        rgb_img = cv2.imread(path)
        # print (path)
        # normal_gt, mask_gt = dataset.get_normal_image(image_id)
        # if len(np.unique(mask_gt)) == 1:
        #     continue
        # Getting the normal gt
        # normal_gt = normal_gt.permute(1, 2, 0).numpy()
        # output_normal_gt = torch.from_numpy(normal_gt)
        # normalized_normal_image_gt = torch.nn.functional.normalize(output_normal_gt, dim=2)
        # normalized_normal_image_gt = normalized_normal_image_gt.double()

        top_predictions = select_top_predictions(mask_pred)
        masks = top_predictions.get_field("mask").numpy()

        rgb_img = np.zeros((480, 640, 3), dtype=float)
        # mask_img = np.copy(127.5 + 127.5 * normal_gt)
        rgb_img = np.asarray(rgb_img, dtype=float)
        print('rgb_img.shape: ', rgb_img.shape)
        print('masks.shape: ', masks.shape)
        for mask in masks:
            index_mask = mask
            print(index_mask.sum())
            rgb_img[index_mask] = np.array([np.random.randint(low=0, high=255),
                                                   np.random.randint(low=0, high=255),
                                                   np.random.randint(low=0, high=255)])

        # for planeIdx in range(1, len(np.unique(masks))):
        #     index_mask =
        #     rgb_img[masks == planeIdx] = np.array([np.random.randint(low=0, high=255),
        #                                                      np.random.randint(low=0, high=255),
        #                                                      np.random.randint(low=0, high=255)])
        #
        # # print(os.getcwd())
        # # # os.makedirs('./plane_normal')
        if not cv2.imwrite('plane_normal/%06d-mask_pred.jpg' % image_id, rgb_img):
            print('saving failed')

        # mask_gt_img = np.copy(127.5 + 127.5 * normal_gt)
        # mask_gt_img = np.asarray(mask_gt_img, dtype=float)
        # for index in range(1, len(np.unique(mask_gt))):
        #     mask_gt_img[mask_gt == index] *= 0.5
        #     mask_gt_img[mask_gt == index] += 0.5 * np.array([np.random.randint(low=0, high=255),
        #                                                      np.random.randint(low=0, high=255),
        #                                                      np.random.randint(low=0, high=255)])
        #
        # if not cv2.imwrite('plane_normal/%06d-mask_gt.jpg' % image_id, mask_gt_img):
        #     print('saving failed')

def select_top_predictions(predictions):
    """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > 0.9).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]
