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
        image_size=480,
    ):
        
        self.root = Path(root)
        image_paths = self._get_image_paths()
        self.data = [self._get_image(image_path) for image_path in image_paths]

        self.transforms = A.Compose([
            A.Resize(image_size, image_size),
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
    data_root = '/data/KvasirSEG'
    image_size = 480
    batch_size = 4

    dataset = MyDataset(
        root=data_root,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        #drop_last=True,
    )


    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("Mean: {}, Std: {}".format(mean, std))


if __name__ == "__main__":
    main()