import numpy as np
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset



class MyDataset(Dataset):

    def __init__(
        self,
        root='data/PH2',
    ):
        
        self.root = Path(root)
        image_paths = self._get_image_paths()
        self.data = [self._get_image(image_path) for image_path in image_paths]

        self.transforms = A.Compose([
            A.Normalize(0, 1),
            ToTensorV2(),
        ])

    def _get_image_paths(self):
        root_image = Path(self.root / 'images')
        image_paths = root_image.rglob('*.[btjp][mipn][pfjg]')
        image_paths = sorted(image_paths)

        return image_paths

    def _get_image(self, image_path):
        image = cv2.imread(image_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
        
    def __getitem__(self, index):
        image = self.data[index]
        image = self.transforms(image=image)['image']

        return image

    def __len__(self):
        return len(self.data)
    


def main():
    data_root = '/mnt/Workspace/hebbian-skin-cancer-segmentation/data/TREND'

    dataset = MyDataset(
        root=data_root,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        #drop_last=True,
    )


    mean_r, mean_g, mean_b = 0., 0., 0.
    std_r, std_g, std_b = 0., 0., 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.shape[0]
        mean_r +=  torch.mean(data[0, 0])
        mean_g +=  torch.mean(data[0, 1])
        mean_b +=  torch.mean(data[0, 2])
        std_r +=  torch.std(data[0, 0])
        std_g +=  torch.std(data[0, 1])
        std_b +=  torch.std(data[0, 2])
        nb_samples += batch_samples

    mean = [mean_r / nb_samples, mean_g / nb_samples, mean_b / nb_samples]
    std = [std_r / nb_samples, std_g / nb_samples, std_b / nb_samples]

    print("Mean: {}, Std: {}".format(mean, std))


if __name__ == "__main__":
    main()