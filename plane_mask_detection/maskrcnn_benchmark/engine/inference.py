# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from pytorch_local.maskrcnn_benchmark.config import cfg
from pytorch_local.maskrcnn_benchmark.data.datasets.evaluation import evaluate
from pytorch_local.maskrcnn_benchmark.data.datasets.evaluation import evaluate_normal
from pytorch_local.maskrcnn_benchmark.data.datasets.evaluation import _evaluate_normal
from pytorch_local.maskrcnn_benchmark.data.datasets.evaluation import evaluate_plane_normal
from pytorch_local.maskrcnn_benchmark.data.datasets.evaluation import evaluate_plane_normal_consistency

from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    normal_error_results_dict = {}
    normal_results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, normal_images, mask_images, image_ids, gravity_dirs, gravity_masks = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                # print('gravity_dirs:', gravity_dirs)
                if gravity_dirs is not None:
                    output, normal_error, normal_pred = model(images.to(device),
                                                        normal_images=normal_images.to(device),
                                                        mask_images=mask_images.to(device),
                                                        gravity_dirs=gravity_dirs.to(device),
                                                        gravity_masks=gravity_masks.to(device))
                else:
                    output, normal_error, normal_pred = model(images.to(device),
                                                              normal_images=normal_images.to(device),
                                                              mask_images=mask_images.to(device))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]

        results_dict.update(
           {img_id: result for img_id, result in zip(image_ids, output)}
        )
        normal_error_results_dict.update(
            {img_id: result.detach().cpu() for img_id, result in zip(image_ids, normal_error)}
        )
        normal_results_dict.update(
            {img_id: result.detach().cpu() for img_id, result in zip(image_ids, normal_pred)}
        )
    return results_dict, normal_error_results_dict, normal_results_dict
    #return normal_error_results_dict
    # return normal_error_results_dict, normal_results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    # if len(image_ids) != image_ids[-1] + 1:
    #     logger = logging.getLogger("maskrcnn_benchmark.inference")
    #     logger.warning(
    #         "Number of images that were gathered from multiple processes is not "
    #         "a contiguous set. Some images might be missing from the evaluation"
    #     )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        visualization=True
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions, normal_error, normal_predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    #normal_error = compute_on_dataset(model, data_loader, device, inference_timer)
    # normal_error, normal_predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    normal_error = _accumulate_predictions_from_multiple_gpus(normal_error)
    normal_predictions = _accumulate_predictions_from_multiple_gpus(normal_predictions)

    # if visualization:
    #     normal_predictions = _accumulate_predictions_from_multiple_gpus(normal_predictions)
    # else:
    #     normal_predictions = None
    if not is_main_process():
        return

    #if output_folder:
    #    torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    # return _evaluate_normal(dataset, normal_error, normal_predictions, output_folder)
    return evaluate_plane_normal(dataset=dataset,
                                             predictions=predictions,
                                             normal_predictions=normal_predictions,
                                             output_folder=output_folder)
    # return evaluate_plane_normal_consistency(dataset=dataset,
    #                                         normal_predictions=normal_predictions,
    #                                         output_folder=output_folder)

    # return evaluate(dataset=dataset,
    #                 predictions=predictions,
    #                 output_folder=output_folder,
    #                 **extra_args), \
    #        evaluate_normal(dataset=dataset,
    #                        predictions=normal_error,
    #                        output_folder=output_folder), \
    #        evaluate_plane_normal_consistency(dataset=dataset,
    #                                          predictions=predictions,
    #                                          normal_predictions=normal_predictions,
    #                                          output_folder=output_folder)
