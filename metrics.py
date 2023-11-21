import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import hd95, assd


def _dice_jaccard_single_class(y_true, y_pred, smooth, axis):
    intersection = (y_true * y_pred).sum(axis)
    sum_ = y_true.sum(axis) + y_pred.sum(axis)
    union = sum_ - intersection

    jaccard = (intersection + smooth) / (union + smooth)
    dice = 2. * (intersection + smooth) / (sum_ + smooth)

    return dice.mean(), jaccard.mean()


def _hausdorff_distance_single_class(y_true, y_pred):
    hausdorff_distance = hd95(y_pred, y_true)

    return hausdorff_distance


def _average_surface_distance_single_class(y_true, y_pred):
    average_surface_distance = assd(y_pred, y_true)

    return average_surface_distance


def _atleast_nhwc(x):
    if x.ndim == 2:
        x = x[None, ..., None]
    elif x.ndim == 3:
        x = x[None, ...]
        
    return x


def dice_jaccard(y_true, y_pred, smooth=1, thr=None, prefix=''):
    """
    Computes Dice and Jaccard coefficients.

    Args:
        y_true (ndarray): (H,W,C)-shaped groundtruth map with binary values (0, 1)
        y_pred (ndarray): (H,W,C)-shaped predicted map with values in [0, 1]
        smooth (int, optional): Smoothing factor to avoid ZeroDivisionError. Defaults to 1.
        thr (float, optional): Threshold to binarize predictions; if None, the soft version of
                               the coefficients are computed. Defaults to None.

    Returns:
        dict: computed metrics organized with the following keys
          - segm/{dice,jaccard}: Micro-averaged Dice and Jaccard coefficients.
          - ...
    """
    y_pred = _atleast_nhwc(y_pred)
    y_true = _atleast_nhwc(y_true)

    y_pred = (y_pred >= thr) if thr is not None else y_pred

    micro_dice, micro_jaccard = _dice_jaccard_single_class(y_true, y_pred, smooth, axis=(1, 2, 3))

    metrics = {
        f'segm/{prefix}dice': micro_dice.item(),
        f'segm/{prefix}jaccard': micro_jaccard.item(),
    }
    
    return metrics


def hausdorff_distance(y_true, y_pred, thr=None, prefix=''):
    """
    Computes Hausdorff Distance (95HD).

    Args:
        y_true (ndarray): (H,W,C)-shaped groundtruth map with binary values (0, 1)
        y_pred (ndarray): (H,W,C)-shaped predicted map with values in [0, 1]

    Returns:
        dict: computed metrics organized with the following keys
          - segm/95hd: Hausdorff distance.
          - ...
    """
    y_pred = _atleast_nhwc(y_pred)
    y_true = _atleast_nhwc(y_true)

    y_pred = (y_pred >= thr) if thr is not None else y_pred

    if torch.any(y_pred) and torch.any(y_true):
        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()
        hausdorff_distance = _hausdorff_distance_single_class(y_true, y_pred).item()
    else:
        hausdorff_distance = None

    metrics = {
        f'segm/{prefix}95hd': hausdorff_distance,
    }
    
    return metrics


def average_surface_distance(y_true, y_pred, thr=None, prefix=''):
    """
    Computes Average Surface Distace (ASD).

    Args:
        y_true (ndarray): (H,W,C)-shaped groundtruth map with binary values (0, 1)
        y_pred (ndarray): (H,W,C)-shaped predicted map with values in [0, 1]

    Returns:
        dict: computed metrics organized with the following keys
          - segm/asd: average surface distance.
          - ...
    """
    y_pred = _atleast_nhwc(y_pred)
    y_true = _atleast_nhwc(y_true)

    y_pred = (y_pred >= thr) if thr is not None else y_pred

    if torch.any(y_pred) and torch.any(y_true):
        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()
        average_surface_distance = _average_surface_distance_single_class(y_true, y_pred).item()
    else:
        average_surface_distance = None

    metrics = {
        f'segm/{prefix}asd': average_surface_distance,
    }
    
    return metrics


class ElboMetric:

    def __init__(self, beta=1):
        self.beta = beta
    
    def __call__(self, outputs, targets):
        reconstr = outputs['reconstr']
        mu = outputs['mu']
        log_var = outputs['log_var']
        reconstr_loss = F.mse_loss(reconstr, targets)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        return reconstr_loss + self.beta * kld_loss
