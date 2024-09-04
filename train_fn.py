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
from denoising_diffusion_pytorch import GaussianDiffusion

from utils import wavelet_filtering, update_teacher_variables, superpix_segment
from metrics import dice_jaccard, hausdorff_distance, average_surface_distance, EntropyMetric

tqdm = partial(tqdm, dynamic_ncols=True)

# creating logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _save_image_and_segmentation_maps(image, image_id, segmentation_map, target_map, cfg, split="validation", outdir=None, h_image=None, l_image=None, reconstructed_image=None, perturbed_segmentation_map=None, segm_threshold=None):
    debug_folder_name = 'train_output_debug' if split == "train" else 'val_output_debug' if split == "validation" else Path(outdir / "test_output_debug")
    debug_dir = Path(debug_folder_name)
    debug_dir.mkdir(parents=True, exist_ok=True)

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

    if h_image is not None and l_image is not None:
        image_name, image_ext = image_id.as_posix().rsplit(".", 1)
        h_image_name, l_image_name = Path("{}_h.{}".format(image_name, image_ext)), Path("{}_l.{}".format(image_name, image_ext))
        _scale_and_save(h_image, debug_dir / h_image_name, denormalize_image=True)
        _scale_and_save(l_image, debug_dir / l_image_name, denormalize_image=True)
    
    if reconstructed_image is not None:
        image_name, image_ext = image_id.as_posix().rsplit(".", 1)
        r_image_name = Path("{}_reconstructr.{}".format(image_name, image_ext))
        _scale_and_save(reconstructed_image, debug_dir / r_image_name, denormalize_image=True)
    
    n_classes = segmentation_map.shape[0]
    segmentation_map = (segmentation_map >= segm_threshold) if segm_threshold is not None else segmentation_map
    for i in range(n_classes):
        _scale_and_save(segmentation_map[i, :, :], debug_dir / f'{image_id.stem}_segm_cls{i}.png')
        _scale_and_save(target_map[i, :, :], debug_dir / f'{image_id.stem}_target_cls{i}.png')
        if perturbed_segmentation_map is not None:
            _scale_and_save(perturbed_segmentation_map[i, :, :], debug_dir / f'{image_id.stem}_perturbed_segm_cls{i}.png')

def _save_debug_metrics(metrics, epoch):
    debug_dir = Path('output_debug')
    debug_dir.mkdir(exist_ok=True)
    
    metrics = pd.DataFrame(metrics)
    csv_file_path = 'validation_metrics_epoch_{}.csv'.format(epoch).format(epoch)
    metrics.to_csv(debug_dir / Path(csv_file_path), index=False)


def train_one_epoch(dataloader, model, optimizer, device, writer, epoch, cfg, teacher_model=None):
    """ Trains the model for one epoch. """
    model.train()
    optimizer.zero_grad()

    if cfg.optim.diffusion_timestamp != 0:
        diffusion = GaussianDiffusion(
            model.net,
            image_size = cfg.data.image_size,
            timesteps = cfg.optim.diffusion_timestamp,    # number of steps
            device = device,
        )
    else:
        criterion = hydra.utils.instantiate(cfg.optim.loss)

    if cfg.optim.entropy_lambda == 'adaptive':
        entropy_lambda = cfg.optim.starting_entropy_lambda * ((epoch+1) / cfg.optim.epochs)
    else:
        entropy_lambda = cfg.optim.entropy_lambda

    if cfg.optim.wavelet_lambda == 'adaptive':
        wavelet_lambda = cfg.optim.starting_wavelet_lambda * ((epoch+1) / cfg.optim.epochs)
    else:
        wavelet_lambda = cfg.optim.wavelet_lambda

    if cfg.optim.perturbation_lambda == 'adaptive':
        perturbation_lambda = cfg.optim.starting_perturbation_lambda * ((epoch+1) / cfg.optim.epochs)
    else:
        perturbation_lambda = cfg.optim.perturbation_lambda

    if cfg.optim.teacher_lambda == 'adaptive':
        teacher_lambda = cfg.optim.starting_teacher_lambda * ((epoch+1) / cfg.optim.epochs)
    else:
        teacher_lambda = cfg.optim.teacher_lambda 

    if cfg.optim.superpix_lambda == 'adaptive':
        superpix_lambda = cfg.optim.starting_superpix_lambda * ((epoch+1) / cfg.optim.epochs)
    else:
        superpix_lambda = cfg.optim.superpix_lambda 

    if cfg.optim.diffusion_timestamp == 0:
        entropy_cost = EntropyMetric() if entropy_lambda != 0 else None
        aux_criterion = torch.nn.BCEWithLogitsLoss() if criterion.__class__.__name__ == 'ElboMetric' else None

    metrics = []
    n_batches = len(dataloader)
    progress = tqdm(dataloader, desc='TRAIN', leave=False)
    for i, sample in enumerate(progress):
        images, labels, image_ids, _ = sample
        batch_size = labels.shape[0]
        if labels.ndim < 4: labels = labels.unsqueeze(dim=1)    # Unsqueeze channel dimension if necessary

        images, labels = images.to(device), labels.to(device)
        visible_labels = torch.all(labels.view(labels.shape[0], -1) != -1, dim=1)
        any_visible_label = torch.any(visible_labels)

        if wavelet_lambda != 0:
            h_images, l_images = wavelet_filtering(images)
            unfiltered_images = images
            images = torch.cat([h_images, l_images], dim=0)

        if teacher_lambda != 0:
            noised_images = torch.clamp(torch.randn_like(images[~visible_labels]) * 0.1, -0.2, 0.2) + images[~visible_labels]

        # computing outputs
        if cfg.optim.diffusion_timestamp != 0:
            total_loss = diffusion(images)
        else:
            output = model(images)
            preds = output['output'] if isinstance(output, dict) else output

            preds_prob = (torch.sigmoid(preds)) if criterion.__class__.__name__.endswith("WithLogitsLoss") or criterion.__class__.__name__ == "ElboMetric" else preds

            if wavelet_lambda != 0:
                h_preds, l_preds = preds[:preds.shape[0]//2], preds[preds.shape[0]//2:]
                h_preds_prob, l_preds_prob = preds_prob[:preds_prob.shape[0]//2], preds_prob[preds_prob.shape[0]//2:]
                preds_prob = h_preds_prob  if cfg.optim.use_h_wavelet else l_preds_prob
                preds = h_preds if cfg.optim.use_h_wavelet else l_preds
                max_l_preds_prob = (l_preds_prob > 0.5).float()
                max_h_preds_prob = (h_preds_prob > 0.5).float()

            if perturbation_lambda != 0:
                perturbation_preds = preds
                perturbation_preds_prob = preds_prob
                preds = preds[:batch_size]
                preds_prob = preds_prob[:batch_size]

            if teacher_lambda != 0 and teacher_model is not None:
                with torch.no_grad():
                    teacher_output = teacher_model(noised_images)
                    teacher_preds = teacher_output['output'] if isinstance(teacher_output, dict) else teacher_output
                    # teacher_prob = (torch.sigmoid(teacher_preds)) if criterion.__class__.__name__.endswith("WithLogitsLoss") or criterion.__class__.__name__ == "ElboMetric" else teacher_preds

            if superpix_lambda != 0:
                superpix_labels = -torch.ones_like(labels)
                if torch.any(~visible_labels): superpix_labels[~visible_labels] = superpix_segment(images[~visible_labels])

            # computing loss and backwarding it
            if aux_criterion is not None:
                aux_loss = aux_criterion(preds[visible_labels], labels[visible_labels]) if any_visible_label else torch.zeros(1, dtype=preds.dtype, device=device, requires_grad=True)
                aux_loss.backward(retain_graph=True)
                if hasattr(model, 'reset_internal_grads'): model.reset_internal_grads()
            else:
                aux_loss = torch.zeros(1, dtype=preds.dtype, device=device, requires_grad=True)
            if criterion.__class__.__name__ != 'ElboMetric':
                loss = criterion(preds[visible_labels], labels[visible_labels]) if any_visible_label else torch.zeros(1, dtype=preds.dtype, device=device, requires_grad=True)
                entropy_loss = entropy_cost(preds[~visible_labels]) if entropy_lambda != 0 and torch.any(~visible_labels) else torch.zeros(1, dtype=preds.dtype, device=device, requires_grad=True)
                wavelet_loss = criterion(h_preds[~visible_labels], max_l_preds_prob[~visible_labels]) + criterion(l_preds[~visible_labels], max_h_preds_prob[~visible_labels]) if wavelet_lambda != 0 and torch.any(~visible_labels) else torch.zeros(1, dtype=preds.dtype, device=device, requires_grad=True)
                perturbation_loss = torch.zeros(1, dtype=preds.dtype, device=device, requires_grad=True)
                for i in range(cfg.optim.perturbation):
                    perturbation_loss = perturbation_loss + (torch.mean((perturbation_preds[:batch_size][~visible_labels] - perturbation_preds[(i+1)*batch_size:(i+2)*batch_size][~visible_labels])**2) if torch.any(~visible_labels) else 0)
                teacher_loss = torch.mean((preds[~visible_labels] - teacher_preds)**2) if teacher_lambda != 0 else torch.zeros(1, dtype=preds.dtype, device=device, requires_grad=True)
                superpix_loss = criterion(preds[~visible_labels], superpix_labels[~visible_labels]) if torch.any(~visible_labels) else torch.zeros(1, dtype=preds.dtype, device=device, requires_grad=True)
            else:
                loss = criterion(output, images)
                entropy_loss = torch.zeros(1, dtype=preds.dtype, device=device, requires_grad=True)
                wavelet_loss = torch.zeros(1, dtype=preds.dtype, device=device, requires_grad=True)
                perturbation_loss = torch.zeros(1, dtype=preds.dtype, device=device, requires_grad=True)
                teacher_loss = torch.zeros(1, dtype=preds.dtype, device=device, requires_grad=True)
                superpix_loss = torch.zeros(1, dtype=preds.dtype, device=device, requires_grad=True)

            total_loss = loss + entropy_lambda * entropy_loss + wavelet_lambda * wavelet_loss + perturbation_lambda * perturbation_loss + teacher_lambda * teacher_loss + superpix_lambda * superpix_loss
        
        total_loss.backward()
        
        if cfg.optim.diffusion_timestamp == 0:
            # NCHW -> NHWC
            coefs_pixel = dice_jaccard(labels[visible_labels].movedim(1, -1), preds_prob[visible_labels].movedim(1, -1)) if any_visible_label else {'segm/dice': None, 'segm/jaccard': None}
            coefs_hausdorff_distance = hausdorff_distance(labels[visible_labels].movedim(1, -1), preds_prob[visible_labels].movedim(1, -1), thr=0.5) if any_visible_label else {'segm/95hd': None}
            coefs_average_surface_distance = average_surface_distance(labels[visible_labels].movedim(1, -1), preds_prob[visible_labels].movedim(1, -1), thr=0.5) if any_visible_label else {'segm/asd': None}
            batch_metrics = {
                'loss ' + '(BCE)' if criterion.__class__.__name__ != 'ElboMetric' else '(ELBO)': loss.item(),
                'aux_loss (BCE)': aux_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'wavelet_loss': wavelet_loss.item(),
                'perturbation_loss': perturbation_loss.item(),
                'teacher_loss': teacher_loss.item(),
                'superpix_loss': superpix_loss.item(),
                'dice': coefs_pixel['segm/dice'],
                'jaccard': coefs_pixel['segm/jaccard'],
                'hausdorff distance': coefs_hausdorff_distance['segm/95hd'],
                'average surface distance': coefs_average_surface_distance['segm/asd'],
            }
            metrics.append(batch_metrics)

            postfix = {metric: f'{value:.3f}' if value is not None else None for metric, value in batch_metrics.items()}
            progress.set_postfix(postfix)

            if (i + 1) % cfg.optim.batch_accumulation == 0 or (i + 1) == n_batches:
                if hasattr(model, 'local_update'): model.local_update()
                optimizer.step()
                optimizer.zero_grad()
                if teacher_model is not None:
                    update_teacher_variables(model, teacher_model, cfg.optim.teacher_alpha, epoch)

            if (i + 1) % cfg.optim.log_every == 0:
                batch_metrics.update({'lr': optimizer.param_groups[0]['lr']})
                n_iter = epoch * n_batches + i
                for metric, value in batch_metrics.items():
                    if value is not None:
                        writer.add_scalar(f'train/{metric}', value, n_iter)
                writer.add_scalar('train/entropy_lambda', entropy_lambda, n_iter)
                writer.add_scalar('train/wavelet_lambda', wavelet_lambda, n_iter)
                writer.add_scalar('train/perturbation_lambda', perturbation_lambda, n_iter)
                writer.add_scalar('train/teacher_lambda', teacher_lambda, n_iter)
                writer.add_scalar('train/superpix_lambda', superpix_lambda, n_iter)

            if cfg.optim.debug and epoch % cfg.optim.debug_freq == 0 and cfg.optim.save_debug_train_images:
                if wavelet_lambda != 0:
                    for unfiltered_image, h_image, l_image, label, image_id, pred_seg_map in zip(unfiltered_images, h_images, l_images, labels, image_ids, preds_prob):
                        _save_image_and_segmentation_maps(unfiltered_image, image_id, pred_seg_map, label, cfg, split="train", h_image=h_image, l_image=l_image)
                elif criterion.__class__.__name__ == 'ElboMetric':
                    for image, reconstr_image, label, image_id, pred_seg_map in zip(images, output['reconstr'], labels, image_ids, preds_prob):
                        _save_image_and_segmentation_maps(image, image_id, pred_seg_map, label, cfg, split="train", reconstructed_image=reconstr_image)
                elif perturbation_lambda != 0:
                    for image, label, image_id, pred_seg_map, first_perturbed_pred_seg_map in zip(images, labels, image_ids, perturbation_preds_prob[:batch_size], perturbation_preds_prob[batch_size:2*batch_size]):
                        _save_image_and_segmentation_maps(image, image_id, pred_seg_map, label, cfg, split="train", perturbed_segmentation_map=first_perturbed_pred_seg_map)
                elif superpix_lambda != 0:
                    for image, label, superpix_label, image_id, pred_seg_map in zip(images, labels, superpix_labels, image_ids, preds_prob):
                        _save_image_and_segmentation_maps(image, image_id, pred_seg_map, label, cfg, split="train", perturbed_segmentation_map=superpix_label)
                else:
                    for image, label, image_id, pred_seg_map in zip(images, labels, image_ids, preds_prob):
                        _save_image_and_segmentation_maps(image, image_id, pred_seg_map, label, cfg, split="train")

        else:
            batch_metrics = {
                'loss diffusion': total_loss.item(),
            }
            metrics.append(batch_metrics)
            
            postfix = {metric: f'{value:.3f}' if value is not None else None for metric, value in batch_metrics.items()}
            progress.set_postfix(postfix)
            
            if (i + 1) % cfg.optim.batch_accumulation == 0 or (i + 1) == n_batches:
                optimizer.step()
                optimizer.zero_grad()

            if (i + 1) % cfg.optim.log_every == 0:
                batch_metrics.update({'lr': optimizer.param_groups[0]['lr']})
                n_iter = epoch * n_batches + i
                for metric, value in batch_metrics.items():
                    if value is not None:
                        writer.add_scalar(f'train/{metric}', value, n_iter)

    metrics = pd.DataFrame(metrics).mean(axis=0).to_dict()

    return metrics


@torch.no_grad()
def validate(dataloader, model, device, epoch, cfg):
    """ Evaluate model on validation data. """
    model.eval()
    validation_device = cfg.optim.val_device
    criterion = hydra.utils.instantiate(cfg.optim.loss)
    aux_criterion = torch.nn.BCEWithLogitsLoss() if criterion.__class__.__name__ == 'ElboMetric' else None

    use_waivelet_filtering = cfg.optim.wavelet_lambda != 0

    metrics = []

    n_images = len(dataloader)
    progress = tqdm(dataloader, total=n_images, desc='EVAL', leave=False)

    for i, sample in enumerate(progress):
        images, labels, image_ids, _ = sample

        if use_waivelet_filtering:
            h_images, l_images = wavelet_filtering(images)
            unfiltered_images = images
            images = h_images if cfg.optim.use_h_wavelet else l_images

        # Un-batching
        for i, (image, label, image_id) in enumerate(zip(images, labels, image_ids)):
            image, label = torch.unsqueeze(image, dim=0).to(validation_device), label[None, None, ...].to(validation_device)

            # computing outputs
            output = model(image)
            pred = output['output'] if isinstance(output, dict) else output
            pred_prob = (torch.sigmoid(pred)) if criterion.__class__.__name__.endswith("WithLogitsLoss") or criterion.__class__.__name__ == "ElboMetric" else pred
 
            # computing metrics
            if aux_criterion is not None:
                aux_loss = aux_criterion(pred, label)
            else:
                aux_loss = torch.zeros(1, dtype=pred.dtype, device=device, requires_grad=True)
            if criterion.__class__.__name__ != 'ElboMetric':
                loss = criterion(pred, label)
            else:
                loss = criterion(output, image)
            segm_metrics_pixel, segm_metrics_distance = {}, {}
            segm_metrics_pixel = dice_jaccard(label.movedim(1, -1), pred_prob.movedim(1, -1))    # NCHW -> NHWC
            segm_metrics_distance = hausdorff_distance(label.movedim(1, -1), pred_prob.movedim(1, -1), thr=0.5)
            segm_metrics_distance.update(average_surface_distance(label.movedim(1, -1), pred_prob.movedim(1, -1), thr=0.5))

            metrics.append({
                'image_id': image_id,
                'segm/loss (BCE)' if criterion.__class__.__name__ != 'ElboMetric' else 'reconstr/loss (ELBO)': loss.item(),
                'segm/aux_loss (BCE)': aux_loss.item(),
                **segm_metrics_pixel,
                **segm_metrics_distance,
            })

            if cfg.optim.debug and epoch % cfg.optim.debug_freq == 0 and cfg.optim.save_debug_val_images:
                if use_waivelet_filtering:
                    _save_image_and_segmentation_maps(unfiltered_images[i], image_id, pred_prob, label, cfg, split="validation", h_image=h_images[i], l_image=l_images[i])
                elif criterion.__class__.__name__ == 'ElboMetric':
                    _save_image_and_segmentation_maps(image, image_id, pred_prob, label, cfg, split="validation", reconstructed_image=output['reconstr'][0])
                else:
                    _save_image_and_segmentation_maps(image, image_id, pred_prob, label, cfg, split="validation")  

    if cfg.optim.debug:
         _save_debug_metrics(metrics, epoch)

    metrics = pd.DataFrame(metrics).set_index('image_id')
    metrics = metrics.mean(axis=0, skipna=True).to_dict()

    return metrics


@torch.no_grad()
def predict(dataloader, model, device, cfg, outdir, debug=0, csv_file_name='preds.csv', segm_threshold=None):
    """ Make predictions on data. """
    model.eval()
    criterion = hydra.utils.instantiate(cfg.optim.loss)

    use_waivelet_filtering = cfg.optim.wavelet_lambda != 0

    metrics = []

    n_images = len(dataloader)
    progress = tqdm(dataloader, total=n_images, desc='EVAL', leave=False)

    for i, sample in enumerate(progress):
        images, labels, image_ids, _ = sample

        if use_waivelet_filtering:
            h_images, l_images = wavelet_filtering(images)
            unfiltered_images = images
            images = h_images if cfg.optim.use_h_wavelet else l_images

        # Un-batching
        for i, (image, label, image_id) in enumerate(zip(images, labels, image_ids)):
            image, label = torch.unsqueeze(image, dim=0).to(device), label[None, None, ...].to(device)

            # computing outputs
            output = model(image)
            pred = output['output'] if isinstance(output, dict) else output
            pred_prob = (torch.sigmoid(pred)) if criterion.__class__.__name__.endswith("WithLogitsLoss") or criterion.__class__.__name__ == "ElboMetric" else pred

            # computing metrics
            segm_metrics_pixel, segm_metrics_distance = {}, {}
            segm_metrics_pixel = dice_jaccard(label.movedim(1, -1), pred_prob.movedim(1, -1), thr=segm_threshold)    # NCHW -> NHWC
            segm_metrics_distance = hausdorff_distance(label.movedim(1, -1), pred_prob.movedim(1, -1), thr=0.5)
            segm_metrics_distance.update(average_surface_distance(label.movedim(1, -1), pred_prob.movedim(1, -1), thr=0.5))

            metrics.append({
                'image_id': image_id,
                **segm_metrics_pixel,
                **segm_metrics_distance,
            })

            if outdir and debug:
                if use_waivelet_filtering:
                    _save_image_and_segmentation_maps(unfiltered_images[i], image_id, pred_prob, label, cfg, split="test", outdir=outdir, h_image=h_images[i], l_image=l_images[i], segm_threshold=segm_threshold)
                elif criterion.__class__.__name__ == 'ElboMetric':
                    _save_image_and_segmentation_maps(image, image_id, pred_prob, label, cfg, split="test", outdir=outdir, reconstructed_image=output['reconstr'][0], segm_threshold=segm_threshold)
                else:
                    _save_image_and_segmentation_maps(image, image_id, pred_prob, label, cfg, split="test", outdir=outdir, segm_threshold=segm_threshold)

    metrics = pd.DataFrame(metrics).set_index('image_id')
    metrics = pd.DataFrame(metrics)

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(outdir / csv_file_name)
