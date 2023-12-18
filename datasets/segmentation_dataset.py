import cv2
from pathlib import Path
from tqdm import tqdm
from functools import partial
import random
import math
import numpy as np

import torch
from torch.utils.data import Dataset

tqdm = partial(tqdm, dynamic_ncols=True)


class SegmentationDataset(Dataset):
    """ """

    def __init__(
        self,
        root='data/PH2',
        split='all',
        split_seed=None,
        cross_val_num_buckets=5,
        cross_val_bucket_validation_index=0,
        in_memory=True,
        target=None,
        transforms=None,
        smpleff_regime=1.,
        use_pseudolabel=False,
    ):
        """ Dataset constructor.
        Args:
        """
        assert split in ('all', 'train', 'validation'), "Split must be one of ('train', 'validation', 'all')"
        assert split == 'all' or split_seed is not None, "You must supply split_seed when split != 'all'"
        assert split == 'all' or cross_val_num_buckets > cross_val_bucket_validation_index, "Cross Validation bucket index must be lower than total number of buckets"

        self.root = Path(root)
        
        self.split = split
        self.split_seed = split_seed
        self.cross_val_num_buckets = cross_val_num_buckets
        self.cross_val_bucket_validation_index = cross_val_bucket_validation_index
        self.target = target
        self.transforms = transforms
        self.smpleff_regime = smpleff_regime
        self.use_pseudolabel = use_pseudolabel

        self.in_memory = in_memory

        # get list of images in the given split
        self.image_paths = self._get_images_in_split()
        self.visible_labels = torch.ones(len(self.image_paths)).bool()
        if self.split == 'train':
            num_imgs = math.ceil(self.smpleff_regime * len(self.image_paths))
            if not self.use_pseudolabel:
                self.image_paths = self.image_paths[:num_imgs]
            self.visible_labels[num_imgs:] = False
        if self.root.stem == 'train' and self.split == 'all':
            num_imgs = math.ceil(self.smpleff_regime * len(self.image_paths))
            random.Random(self.split_seed).shuffle(self.image_paths)    # reproducible shuffle
            if not self.use_pseudolabel:
                self.image_paths = self.image_paths[:num_imgs]
            self.visible_labels[num_imgs:] = False
        
        if in_memory:
            self.data = {}
            self.data['image'] = [self._get_image(image_path) for image_path in self.image_paths]
            if self.target:
                self.data['mask'] = [self._get_mask(Path(str(image_path).replace('images', 'targets'))) for image_path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)
    
    def _get_images_in_split(self):
        root_image = Path(self.root / 'images')
        image_paths = root_image.rglob('*.[btjp][mipn][pfjg]')
        image_paths = sorted(image_paths)

        if self.split == 'all':
            return image_paths

        # preparing splits for n-cross-validation 
        random.Random(self.split_seed).shuffle(image_paths)    # reproducible shuffle
        cross_val_num_bucket_samples = int(len(image_paths) / self.cross_val_num_buckets)
        if self.cross_val_bucket_validation_index == (self.cross_val_num_buckets-1):
            val_image_paths = image_paths[self.cross_val_bucket_validation_index*cross_val_num_bucket_samples:]
        else:
            val_image_paths = image_paths[self.cross_val_bucket_validation_index*cross_val_num_bucket_samples:(self.cross_val_bucket_validation_index*cross_val_num_bucket_samples)+cross_val_num_bucket_samples]
        train_image_paths = list(set(image_paths) - set(val_image_paths))

        if self.split == 'train':
            return train_image_paths
        elif self.split == 'validation':
            return val_image_paths
        
    def _get_image(self, image_path):
        image = cv2.imread(image_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
    
    def _get_mask(self, mask_path):
        mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask[mask > 0] = 1.
        #mask[mask != 255] = 0.
        #mask[mask == 255] = 1.

        return mask
        
    def __getitem__(self, index):
        image_id = self.image_paths[index].parts[-1]
        
        if self.in_memory:
            image = self.data['image'][index]
            if self.target:
                mask = self.data['mask'][index]
        else:
            image = self._get_image(self.image_paths[index])
            if self.target:
                mask = self._get_mask(Path(str(self.image_paths[index]).replace('images', 'targets')))
        
        original_size = tuple(image.shape[:2])

        if self.transforms:
            if self.target:
                transformed = self.transforms(image=image, mask=mask)
                image, mask = transformed['image'], transformed['mask']
            else:
                image = self.transforms(image=image)['image']

        if self.target:
            datum = (image,
                     mask if self.visible_labels[index] else mask.detach().clone().fill_(-1)      # Hide label information in semi-supervised training
                     )
        else:
            datum = (image,)
            
        return datum + (image_id, original_size,)
        
    def __str__(self):
        s = f'{self.__class__.__name__}: ' \
            f'{self.split} split, ' \
            f'{len(self)} images'
        return s



# Test code
def main():
    from torch.utils.data import DataLoader
    import numpy as np
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    image_size = 256

    train_transform = A.Compose([
        A.LongestMaxSize(image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
        A.Flip(),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        #A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        #A.RandomBrightnessContrast(p=0.5),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_dataset_params = {
        'root': "data/DataScienceBowl2018",
        'split': "train",
        'split_seed': 87,
        'cross_val_bucket_validation_index': 0,
        'in_memory': False,
        'target': True,
        'transforms': train_transform,
    }

    batch_size = 1
    num_workers = 0
    collate_fn = None

    debug_dir = Path('datasets/trash')
    debug_dir.mkdir(exist_ok=True)

    train_dataset = SegmentationDataset(**train_dataset_params)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    for counter, sample in enumerate(train_loader):
        if counter == 5:
            break

        images, targets, image_ids, original_sizes = sample
        
        for image, target, image_id, original_height, original_width in zip(images, targets, image_ids, *original_sizes):
            print("Image: {}".format(image_id))
            print("Original Size (w-h): {}-{}".format(original_width, original_height))
            image = np.moveaxis(np.array(image), 0, -1)
            Image.fromarray(image).save(debug_dir / 'img_{}'.format(image_id)) 
            Image.fromarray(np.array(target*255).astype(np.uint8)).save(debug_dir / 'mask_{}'.format(image_id)) 

if __name__ == "__main__":
    main()