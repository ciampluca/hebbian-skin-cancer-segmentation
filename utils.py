import collections
import os
from pathlib import Path
import re
import random
import numpy as np
import math

import torch
import torchvision.transforms.functional as tf

import pywt


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CheckpointManager:

    def __init__(
        self,
        ckpt_dir,
        ckpt_format=None,
        current_best={},
        metric_modes=None
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_format = ckpt_format if ckpt_format else self._default_ckpt_format
        self.current_best = collections.defaultdict(lambda: None, current_best)
        self.metric_modes = metric_modes if metric_modes else self._default_metric_mode
    
    def save(self, ckpt, metrics, epoch):
        for metric, value in metrics.items():
            mode = self.metric_modes(metric)
            if mode == 'ignore':
                continue
            
            cur_best = self.current_best[metric] and self.current_best[metric].get('value', None)
            is_new_best = (cur_best is None and not math.isnan(value)) or ((value < cur_best) if mode == 'min' else (value > cur_best))
            if is_new_best:
                best_metric_ckpt_name = self.ckpt_format(metric)
                best_metric_ckpt_path = self.ckpt_dir / best_metric_ckpt_name
                torch.save(ckpt, best_metric_ckpt_path)     # save ckpt overwriting the old one

                # update current best
                self.current_best[metric] = {'value': value, 'epoch': epoch}

        return dict(self.current_best)
    
    @staticmethod
    def _default_ckpt_format(metric_name):
        metric_name = metric_name.replace('/', '-')
        return f'best_model_metric_{metric_name}.pth'
    
    @staticmethod
    def _default_metric_mode(metric_name):
        if 'loss' in metric_name:
            return 'ignore'
        
        if 'dice' in metric_name or 'jaccard' in metric_name:
            return 'max'
        
        if '95hd' in metric_name:
            return 'ignore' #'min'
        
        if 'asd' in metric_name:
            return 'ignore' #'min'
        
        return 'ignore'


def get_init_param_by_name(param_name, param_dict, cfg, default):
    return param_dict.get(param_name, getattr(cfg, param_name, default))


def wavelet_filtering(images):
    images = tf.rgb_to_grayscale(images)
    images = images.squeeze(dim=1)
    np_images = images.cpu().detach().numpy()
    
    h_images, l_images = [], []
    for image in np_images:
        LL, (LH, HL, HH) = pywt.dwt2(image, "db2")
        h_images.append(HH + HL + LH)
        l_images.append(LL)

    h_images = np.stack(h_images, axis=0)
    l_images = np.stack(l_images, axis=0)

    h_images = torch.from_numpy(h_images).to(images.device).unsqueeze(dim=1).repeat(1, 3, 1, 1)
    l_images = torch.from_numpy(l_images).to(images.device).unsqueeze(dim=1).repeat(1, 3, 1, 1)

    h_images = tf.resize(h_images, [images.shape[1], images.shape[2]], antialias=True)
    l_images = tf.resize(l_images, [images.shape[1], images.shape[2]], antialias=True)

    return h_images, l_images


def update_teacher_variables(model, teacher_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for teacher_param, param in zip(teacher_model.parameters(), model.parameters()):
        teacher_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def initialize_teacher_variables(model, teacher_model):
    for teacher_param, param in zip(teacher_model.parameters(), model.parameters()):
        teacher_param.data.add_(param.data)


def superpix_segment(images, thr=0.01):
    def _pixels_close(im, x, y, x_n, y_n, thr=0.01):
        return (im[:, x, y] - im[:, x_n, y_n]).abs().sum().item() < thr

    superpix = torch.zeros_like(images).sum(dim=1)
    for i, im in enumerate(images):
        queue = []
        x, y = random.randint(0, im.shape[-2]), random.randint(0, im.shape[-1])
        queue.append((x, y))
        explored = torch.zeros_like(superpix[i])
        explored[x, y] = 1
        while len(queue) > 0:
            x, y = queue[0]
            queue = queue[1:] if len(queue) > 1 else []
            superpix[i, x, y] = 1
            neighbors = [(x_n, y_n) for x_n in [max(x-1, 0), x, min(x+1, im.shape[-2]-1)] for y_n in [max(y-1, 0), y, min(y+1, im.shape[-1]-1)] if explored[x_n, y_n].item() == 0 and _pixels_close(im, x, y, x_n, y_n, thr)]
            for x_n, y_n in neighbors: 
                queue.append((x_n, y_n))
                explored[x_n, y_n] = 1
    return superpix.unsqueeze(1)

