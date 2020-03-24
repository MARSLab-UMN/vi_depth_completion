import torch
import torch.nn
import torch.nn.functional as F
import numpy as np


def compute_normal_vectors_loss_l2(norm_gt, pred_normals, mask):
    norm1 = pred_normals[:, 0:3, :, :]

    loss = -torch.sum(F.cosine_similarity(norm1, norm_gt, dim=1)) / torch.sum(mask)

    norm1 = Normalize(norm1)
    angle = torch.acos(torch.clamp(torch.sum(norm1 * norm_gt, dim=1), -1, 1)) / np.pi * 180
    angle = angle.view(mask.shape[0], 1, mask.shape[2], mask.shape[3]) * mask
    angle = torch.sum(angle)

    return loss, angle


def compute_normal_vectors_loss_l1(norm_gt, pred_normals, mask, normalize_prediction=True):
    mask = mask.float()
    loss_func = torch.nn.L1Loss(reduction='sum')
    if normalize_prediction:
        norms = Normalize(pred_normals[:, 0:3, :, :])
    else:
        norms = pred_normals[:, 0:3, :, :]

    angle = torch.acos(torch.clamp(torch.sum(norms * norm_gt, dim=1), -1, 1)) / np.pi * 180
    angle = angle.view(mask.shape[0], 1, mask.shape[2], mask.shape[3]) * mask
    angle = torch.sum(angle)

    num_elements = torch.sum(mask).item()
    loss = loss_func(norms * mask, norm_gt * mask) / num_elements
    return loss, angle

