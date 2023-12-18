import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import albumentations

from datasets import SegmentationDataset


DATAROOT = 'data'
OUTDIR = 'data/zca'
AVAILABLE_DEVICES = ['cpu'] + ['cuda:' + str(i) for i in range(torch.cuda.device_count())]

def save_zca(zca, path):
	zca_dict = {'zca': zca.cpu()}
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(zca_dict, path)

def load_zca(path):
	return torch.load(path)['zca']

def whiten(x, zca):
	return torch.conv2d(x, zca, padding=((zca.shape[-2]-1)//2, (zca.shape[-1]-1)//2))

def compute_zca(dataset, device='cpu', smoothing=1e-3, batch=4, filtershape=(3, 32, 32), imagesize=112):
	dataroot = os.path.join(DATAROOT, dataset) # Path to the dataset folder
	outpath = os.path.join(OUTDIR, dataset + '.pth') # Path to the outputfile where the ZCA matrix will be stored
	
	transforms = albumentations.Compose([
		albumentations.LongestMaxSize(imagesize),
		albumentations.PadIfNeeded(imagesize, imagesize, border_mode=0),
		albumentations.pytorch.ToTensorV2(),
	])
	
	dataset = SegmentationDataset(root=dataroot, transforms=transforms)
	dataloader = DataLoader(dataset, batch_size=batch, shuffle=False)
	
	sum = torch.zeros(filtershape[0], device=device)
	sum_sq = torch.zeros(filtershape[0], device=device)
	count = 0
	print("Computing statistics for dataset \"" + dataroot + "\"...")
	for batch in tqdm(dataset):
		inputs, _ = batch
		inputs = inputs.to(device)
		count += inputs.size(0)
		sum += inputs.mean(dim=(2, 3)).sum(0)
		sum_sq += (inputs**2).mean(dim=(2, 3)).sum(0)
	mean = sum / count
	mean_sq = sum_sq / count
	std = (mean_sq - mean**2)**0.5
	
	filtersize = filtershape[0]*filtershape[1]*filtershape[2]
	cov = torch.zeros((filtersize, filtersize), device=device)
	count = 0
	print("Computing covariance matrix for dataset \"" + dataroot + "\" (this might take a while)...")
	for batch in tqdm(dataloader):
		inputs, _ = batch
		inputs = inputs.to(device)
		inputs = (inputs - mean) / std
		inputs_unf = inputs.unfold(2, filtershape[1], filtershape[1]).unfold(3, filtershape[2], filtershape[2]).permute(0, 2, 3, 1, 4, 5).reshape(-1, filtershape)
		cov += inputs_unf.t().matmul(inputs_unf)
		count += inputs.shape[0]
	cov = cov / (count - 1)
	
	print("Computing SVD of covariance matrix for dataset \"" + dataroot + "\" (this might take a while)...")
	U, S, V = torch.svd(cov)
	
	print("Computing ZCA matrix for dataset \"" + dataroot + "\"...")
	zca = U.matmul(torch.diag((S + smoothing) ** -0.5).matmul(U.t()))
	zca = zca.reshape(*filtershape, zca.shape[1])[:, filtershape[1]//2, filtershape[2]//2, :].reshape(-1, *filtershape)
	
	# Save zca matrix
	save_zca(zca, outpath)
	print("ZCA matrix computed and saved.")

if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='GlaS', help='The dataset you what to compute the ZCA matrix for.')
	parser.add_argument('--device', default='cpu', choices=AVAILABLE_DEVICES, help="The device you want to use for the computation.")
	parser.add_argument('--smoothing', type=float, default=1e-3, help="A number specifying the intensity of smoothing.")
	parser.add_argument('--batch', type=int, default=4, help="A number specifying the intensity of smoothing.")
	parser.add_argument('--filtershape', nargs=3, type=int, default=[3, 32, 32], help="A number specifying the intensity of smoothing.")
	parser.add_argument('--imagesize', type=int, default=112, help="The desired size to reshape inputs.")
	args = parser.parse_args()
	
	compute_zca(args.dataroot, args.device, args.smoothing, args.batch, args.filtershape, args.imagesize)