import numpy as np
import torch
import torch.nn.functional as F


def _dice_jaccard_single_class(y_true, y_pred, smooth, axis):
    intersection = (y_true * y_pred).sum(axis)
    sum_ = y_true.sum(axis) + y_pred.sum(axis)
    union = sum_ - intersection

    jaccard = (intersection + smooth) / (union + smooth)
    dice = 2. * (intersection + smooth) / (sum_ + smooth)

    return dice.mean(), jaccard.mean()


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
