import collections
import os
from pathlib import Path
import re
import random

import numpy as np
import torch


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
            is_new_best = cur_best is None or ((value < cur_best) if mode == 'min' else (value > cur_best))
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
            return 'min'
        
        if 'asd' in metric_name:
            return 'min'
        
        return 'ignore'


def get_init_param_by_name(param_name, param_dict, cfg, default):
    return param_dict.get(param_name, getattr(cfg, param_name, default))
