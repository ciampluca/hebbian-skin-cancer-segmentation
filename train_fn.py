# -*- coding: utf-8 -*-
from functools import partial
import logging
import numpy as np
from pathlib import Path

import torch
import torchvision.transforms.functional as F
import hydra

from tqdm import tqdm
import pandas as pd
from PIL import Image

from metrics import dice_jaccard

tqdm = partial(tqdm, dynamic_ncols=True)

# creating logger
log = logging.getLogger(__name__)


def _save_image_and_segmentation_maps(image, image_id, segmentation_map, target_map, cfg, split="validation", outdir=None):
    debug_folder_name = 'train_output_debug' if split == "train" else 'val_output_debug' if split == "validation" else Path(outdir / "test_output_debug")
    debug_dir = Path(debug_folder_name)
    debug_dir.mkdir(parents=True, exist_ok=True)

    seg_rgb = cfg.optim.loss.__class__.__name__ == 'ElboMetric'
    image_mean = cfg.data.image_mean
    image_std = cfg.data.image_std

    image_id = Path(image_id)

    def _scale_and_save(image, path, denormalize_image=False):
        if denormalize_image:
            image = F.normalize(image, mean=[0., 0., 0.], std=[1/image_std[0], 1/image_std[1], 1/image_std[2]])
            image = F.normalize(image, mean=[-image_mean[0], -image_mean[1], -image_mean[2]], std=[1., 1., 1.])

        dim_to_move = 1 if image.ndim == 4 else 0 if image.ndim == 3 else None
        if dim_to_move is not None:
            image = image.movedim(dim_to_move, -1)
        image = (255 * image.cpu().detach().numpy().squeeze()).astype(np.uint8)
        pil_image = Image.fromarray(image).convert("RGB")
        pil_image.save(path)
        
    _scale_and_save(image, debug_dir / image_id, denormalize_image=True)
    
    if seg_rgb:
         _scale_and_save(segmentation_map.movedim(1, -1), debug_dir / f'{image_id.stem}_reconstr.png', denormalize_image=True)
    else:
        n_classes = segmentation_map.shape[0]
        for i in range(n_classes):
            _scale_and_save(segmentation_map[i, :, :], debug_dir / f'{image_id.stem}_segm_cls{i}.png')
            _scale_and_save(target_map[i, :, :], debug_dir / f'{image_id.stem}_target_cls{i}.png')


def _save_debug_metrics(metrics, epoch):
    debug_dir = Path('output_debug')
    debug_dir.mkdir(exist_ok=True)
    
    metrics = pd.DataFrame(metrics)
    csv_file_path = 'validation_metrics_epoch_{}.csv'.format(epoch).format(epoch)
    metrics.to_csv(debug_dir / Path(csv_file_path), index=False)


def train_one_epoch(dataloader, model, optimizer, device, writer, epoch, cfg):
    """ Trains the model for one epoch. """
    model.train()
    optimizer.zero_grad()

    criterion = hydra.utils.instantiate(cfg.optim.loss)

    metrics = []
    n_batches = len(dataloader)
    progress = tqdm(dataloader, desc='TRAIN', leave=False)
    for i, sample in enumerate(progress):
        images, labels, image_ids, _ = sample
        labels = labels.unsqueeze(dim=1)
        images, labels = images.to(device), labels.to(device)

        # computing outputs
        preds = model(images)
        preds_prob = (torch.sigmoid(preds)) if criterion.__class__.__name__.endswith("WithLogitsLoss") else preds

        # computing loss and backwarding it
        loss = criterion(preds, labels)
        loss.backward()

        # NCHW -> NHWC
        batch_metrics = {'loss': loss.item()}
        if criterion.__class__.__name__ != 'ElboMetric':
            coefs = dice_jaccard(labels.movedim(1, -1), preds_prob.movedim(1, -1))
            batch_metrics = {
                'loss': loss.item(),
                'dice': coefs['segm/dice'],
                'jaccard': coefs['segm/jaccard'],
            }
        metrics.append(batch_metrics)

        postfix = {metric: f'{value:.3f}' for metric, value in batch_metrics.items()}
        progress.set_postfix(postfix)

        if (i + 1) % cfg.optim.batch_accumulation == 0 or (i + 1) == n_batches:
            if hasattr(model, 'local_update'): model.local_update()
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % cfg.optim.log_every == 0:
            batch_metrics.update({'lr': optimizer.param_groups[0]['lr']})
            n_iter = epoch * n_batches + i
            for metric, value in batch_metrics.items():
                writer.add_scalar(f'train/{metric}', value, n_iter)

        if cfg.optim.debug and epoch % cfg.optim.debug_freq == 0 and cfg.optim.save_debug_train_images:
            for image, label, image_id, pred_seg_map in zip(images, labels, image_ids, preds_prob):
                _save_image_and_segmentation_maps(image, image_id, pred_seg_map, label, cfg, split="train")

    metrics = pd.DataFrame(metrics).mean(axis=0).to_dict()

    return metrics


@torch.no_grad()
def validate(dataloader, model, device, epoch, cfg):
    """ Evaluate model on validation data. """
    model.eval()
    validation_device = cfg.optim.val_device
    criterion = hydra.utils.instantiate(cfg.optim.loss)

    metrics = []

    n_images = len(dataloader)
    progress = tqdm(dataloader, total=n_images, desc='EVAL', leave=False)

    for i, sample in enumerate(progress):
        images, labels, image_ids, _ = sample

        # Un-batching
        # TODO not efficient, should be done in parallel
        for image, label, image_id in zip(images, labels, image_ids):
            image, label = torch.unsqueeze(image, dim=0).to(validation_device), label[None, None, ...].to(validation_device)

            # computing outputs
            pred = model(image)
            pred_prob = (torch.sigmoid(pred)) if criterion.__class__.__name__.endswith("WithLogitsLoss") else pred
 
            # threshold-free metrics
            loss = criterion(pred, label)
            segm_metrics = {}
            if criterion.__class__.__name__ != 'ElboMetric':
                segm_metrics = dice_jaccard(label.movedim(1, -1), pred_prob.movedim(1, -1))    # NCHW -> NHWC

            metrics.append({
                'image_id': image_id,
                'segm/loss': loss.item(),
                **segm_metrics,
            })

            if cfg.optim.debug and epoch % cfg.optim.debug_freq == 0 and cfg.optim.save_debug_val_images:
                _save_image_and_segmentation_maps(image, image_id, pred_prob, label, cfg, split="validation")

    if cfg.optim.debug:
         _save_debug_metrics(metrics, epoch)

    metrics = pd.DataFrame(metrics).set_index('image_id')
    metrics = metrics.mean(axis=0).to_dict()

    return metrics


@torch.no_grad()
def predict(dataloader, model, device, cfg, outdir, debug=0, csv_file_name='preds.csv'):
    """ Make predictions on data. """
    model.eval()
    criterion = hydra.utils.instantiate(cfg.optim.loss)

    metrics = []

    n_images = len(dataloader)
    progress = tqdm(dataloader, total=n_images, desc='EVAL', leave=False)

    for i, sample in enumerate(progress):
        images, labels, image_ids, _ = sample

        # Un-batching
        # TODO not efficient, should be done in parallel
        for image, label, image_id in zip(images, labels, image_ids):
            image, label = torch.unsqueeze(image, dim=0).to(device), label[None, None, ...].to(device)

            # computing outputs
            pred = model(image)
            pred_prob = (torch.sigmoid(pred)) if criterion.__class__.__name__.endswith("WithLogitsLoss") else pred

            # threshold-free metrics
            segm_metrics = {}
            if criterion.__class__.__name__ != 'ElboMetric':
                segm_metrics = dice_jaccard(label.movedim(1, -1), pred_prob.movedim(1, -1))    # NCHW -> NHWC

            metrics.append({
                'image_id': image_id,
                **segm_metrics,
            })

            if outdir and debug:
                _save_image_and_segmentation_maps(image, image_id, pred_prob, label, cfg, split="test", outdir=outdir)

    metrics = pd.DataFrame(metrics).set_index('image_id')
    metrics = pd.DataFrame(metrics)

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(outdir / csv_file_name)