import os
import logging
from functools import partial
from pathlib import Path
import re

import hydra
import pandas as pd
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from train_fn import train_one_epoch, validate
from utils import seed_everything, seed_worker, CheckpointManager, initialize_teacher_variables

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
tqdm = partial(tqdm, dynamic_ncols=True)
trange = partial(trange, dynamic_ncols=True)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    from omegaconf import OmegaConf; print(OmegaConf.to_yaml(cfg))
    
    run_path = Path.cwd()
    log.info(f"Run path: {run_path}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    device = torch.device(f'cuda' if cfg.gpu is not None else 'cpu')
    log.info(f"Use device {device} for training")

    # Reproducibility
    seed_everything(cfg.seed)
    torch.set_default_dtype(torch.float32)

    # create tensorboard writer
    writer = SummaryWriter()

    # train dataset and dataloader
    g = torch.Generator()
    g.manual_seed(cfg.seed)     # To guarantee reproducibility
    train_dataset = hydra.utils.instantiate(cfg.data.train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.optim.batch_size, shuffle=True, num_workers=cfg.optim.num_workers, worker_init_fn=seed_worker, generator=g, drop_last=cfg.data.drop_last_batch)
    log.info(f'[TRAIN] {train_dataset}')

    # validation dataset and dataloader
    g = torch.Generator()
    g.manual_seed(cfg.seed)     # To guarantee reproducibility
    valid_dataset = hydra.utils.instantiate(cfg.data.validation)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.optim.val_batch_size, num_workers=cfg.optim.num_workers, worker_init_fn=seed_worker, generator=g)
    log.info(f'[VALID] {valid_dataset}')

    # create model
    torch.hub.set_dir(cfg.model.cache_folder)
    model = hydra.utils.instantiate(cfg.model.module, cfg)
    # move model to device
    model.to(device)
    model_param_string = ', '.join(f'{k}={v}' for k, v in cfg.model.module.items() if not k.startswith('_'))
    log.info(f"[MODEL] {cfg.model.name}({model_param_string})")

    # (eventually) create teacher model for MT method
    teacher_model = None
    if cfg.optim.teacher_lambda != 0:
        teacher_model = hydra.utils.instantiate(cfg.model.module, cfg)
        teacher_model.to(device)
        teacher_model_param_string = ', '.join(f'{k}={v}' for k, v in cfg.model.module.items() if not k.startswith('_'))
        log.info(f"[TEACHER MODEL] {cfg.model.name}({teacher_model_param_string})")

    # build the optimizer
    optimizer = hydra.utils.instantiate(cfg.optim.optimizer, model.parameters())
    scheduler = hydra.utils.instantiate(cfg.optim.lr_scheduler, optimizer)

    # optionally load pre-trained weights
    if cfg.model.pretrained:
        if cfg.model.pretrained.startswith('http://') or cfg.model.pretrained.startswith('https://'):
            pre_trained_model = torch.hub.load_state_dict_from_url(
                cfg.model.pretrained, map_location=device, model_dir=cfg.model.cache_folder)
        elif cfg.model.pretrained.startswith('best://'):
            checkpoint_config_path = str(run_path).replace('_ft', "")
            checkpoint_config_path = re.sub("regime-\d+\.\d+", "regime-1.0", checkpoint_config_path)
            checkpoint_config_path = Path(re.sub("run-\d+", "run-0", checkpoint_config_path))
            exp_name = checkpoint_config_path.parts[-4]
            if "base" not in exp_name:
                new_exp_name = exp_name
                if not exp_name.startswith("hunet"):
                    new_exp_name = new_exp_name.split("-", 1)[1]
                new_exp_name = new_exp_name.replace('unet', 'unet_base')
                checkpoint_config_path = Path(str(checkpoint_config_path).replace(exp_name, new_exp_name))
            best_models_folder = checkpoint_config_path / 'best_models'
            metric_name = cfg.model.pretrained.split('best://', 1)[1]
            ckpt_path = best_models_folder / f'best_model_metric_segm-{metric_name}.pth'
            pre_trained_model = torch.load(ckpt_path, map_location=device)
        elif cfg.model.pretrained.startswith('last://'):
            checkpoint_config_path = str(run_path).replace('_ft', "")
            checkpoint_config_path = re.sub("regime-\d+\.\d+", "regime-1.0", checkpoint_config_path)
            checkpoint_config_path = Path(re.sub("run-\d+", "run-0", checkpoint_config_path))
            exp_name = checkpoint_config_path.parts[-4]
            if "base" not in exp_name:
                new_exp_name = exp_name
                if not exp_name.startswith("hunet"):
                    new_exp_name = new_exp_name.split("-", 1)[1]
                new_exp_name = new_exp_name.replace('unet', 'unet_base')
                checkpoint_config_path = Path(str(checkpoint_config_path).replace(exp_name, new_exp_name))
            ckpt_file_name = cfg.model.pretrained.split('last://', 1)[1]
            ckpt_path = checkpoint_config_path / f'{ckpt_file_name}.pth'
            pre_trained_model = torch.load(ckpt_path, map_location=device)
        else:
            ckpt_path = cfg.model.pretrained
            pre_trained_model = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(pre_trained_model['model'])
        if cfg.model.reset_clf and hasattr(model, 'reset_clf'):
            model.reset_clf(cfg.model.reset_clf)    # Resets final classifier with number of output channels specified in cfg.model.reset_clf
        log.info(f"[PRETRAINED]: {cfg.model.pretrained}")

        if cfg.optim.teacher_lambda != 0:
            initialize_teacher_variables(model, teacher_model)
        
    start_epoch = 0
    best_metrics = {}

    train_log_path = 'train_log.csv'
    valid_log_path = 'valid_log.csv'

    train_log = pd.DataFrame()
    valid_log = pd.DataFrame()

    # optionally resume from a saved checkpoint
    if cfg.optim.resume:
        assert Path('last.pth').exists(), 'Cannot find checkpoint for resuming.'
        checkpoint = torch.load('last.pth', map_location=device)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])

        start_epoch = checkpoint['epoch'] + 1
        best_metrics = checkpoint['best_metrics']

        train_log = pd.read_csv(train_log_path, index_col=0, header=[0,1])
        valid_log = pd.read_csv(valid_log_path, index_col=0, header=[0,1])
        log.info(f"[RESUME] Resuming from epoch {start_epoch}")

    # checkpoint manager
    ckpt_dir = Path('best_models')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_manager = CheckpointManager(ckpt_dir, current_best=best_metrics)

    # Train loop
    log.info(f"Training ...")
    progress = trange(start_epoch, cfg.optim.epochs, initial=start_epoch)
    for epoch in progress:
        # train
        train_metrics = train_one_epoch(train_loader, model, optimizer, device, writer, epoch, cfg, teacher_model=teacher_model)
        scheduler.step()  # update lr scheduler

        # convert for pandas
        train_metrics = {(metric_name, 'value'): value for metric_name, value in train_metrics.items()}
        train_metrics = pd.DataFrame(train_metrics, index=[epoch]).rename_axis('epoch')
        train_log = pd.concat([train_log, train_metrics])
        train_log.to_csv(train_log_path)

        # evaluation
        if (epoch + 1) % cfg.optim.val_freq == 0:
            valid_metrics = validate(valid_loader, model, device, epoch, cfg)

            for metric, value in valid_metrics.items():  # log to tensorboard
                writer.add_scalar(f'valid/{metric}', value, epoch)

            # save only if best on some metric (via CheckpointManager)
            best_metrics = ckpt_manager.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': valid_metrics
            }, valid_metrics, epoch)

            # save last checkpoint for resuming
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_metrics': best_metrics,
            }, 'last.pth')

            # convert for pandas
            valid_metrics = {(metric_name, 'value'): value for metric_name, value in valid_metrics.items()}
            valid_metrics = pd.DataFrame(valid_metrics, index=[epoch]).rename_axis('epoch')
            valid_log = pd.concat([valid_log, valid_metrics])
            valid_log.to_csv(valid_log_path)

    log.info("Training ended. Exiting....")


if __name__ == "__main__":
    main()