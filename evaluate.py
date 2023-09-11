import argparse
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from train_fn import predict


log = logging.getLogger(__name__)



def main(args):
    run_path = Path(args.run)
    hydra_cfg = OmegaConf.load(run_path / '.hydra' / 'hydra.yaml')['hydra']
    OmegaConf.register_new_resolver("hydra", lambda x: OmegaConf.select(hydra_cfg, x))

    cfg = OmegaConf.load(run_path / '.hydra' / 'config.yaml')
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(args.device)
    
    # create test dataset and dataloader
    test_dataset = cfg.data.validation
    test_dataset.root = args.data_root if args.data_root else test_dataset.root
    test_dataset.split = args.test_split
    test_dataset.target = args.load_target
    test_dataset.in_memory = args.in_memory
    test_dataset = hydra.utils.instantiate(test_dataset)

    test_batch_size = cfg.optim.val_batch_size if args.batch_size is None else args.batch_size
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=cfg.optim.num_workers)
    log.info(f'[TEST] {test_dataset}')

    # create model and move to device
    model = hydra.utils.instantiate(cfg.model.module, skip_weights_loading=True)
    model.to(device)
    
    # resume from a saved checkpoint
    best_models_folder = run_path / 'best_models'
    metric_name = args.best_on_metric.replace('/', '-')
    ckpt_path = best_models_folder / f'best_model_metric_segm-{metric_name}.pth'
    log.info(f"[CKPT]: Loading {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # set csv output file and outdir
    outdir = (run_path / 'test_predictions') if args.save else None
        
    # make predictions
    predict(test_loader, model, device, cfg, outdir, debug=args.debug, csv_file_name=args.output_file_name)

    log.info("Evaluation ended. Exiting....")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform evaluation on test set')
    parser.add_argument('run', help='Path to run dir')
    parser.add_argument('-d', '--device', default='cuda', help='device to use for prediction')
    parser.add_argument('--best-on-metric', default='dice', help='select snapshot that optimizes this metric [dice, jaccard, loss]')
    parser.add_argument('--debug', type=int, default=1, help='draw images for debugging')
    parser.add_argument('--data-root', default=None, help='root of the images')
    parser.add_argument('--test-split', default='validation', help='split to be used for evaluation')
    parser.add_argument('--load-target', default=True, help='load also target')
    parser.add_argument('--in-memory', default=False, help='load dataset in memory')
    parser.add_argument('--batch-size', type=int, default=None, help='batch size used for evaluation')
    parser.add_argument('--output-file-name', default='preds.csv', help='file name where predictions will be stored')
    parser.set_defaults(save=True)

    args = parser.parse_args()
    main(args)