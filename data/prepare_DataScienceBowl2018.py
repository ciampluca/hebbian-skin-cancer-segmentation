import os
import numpy as np
from shutil import copyfile

from PIL import Image
from tqdm import tqdm


DATA_ROOT = "DataScienceBowl2018"


def main():
    import sys
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    root_dataset = os.path.join(script_directory, DATA_ROOT)

    for item in tqdm(os.listdir(os.path.join(root_dataset, "train_data"))):
        img_path = os.path.join(root_dataset, "train_data", item, "images", "{}.png".format(item))
        if not os.path.exists(os.path.join(root_dataset, "images")):
            os.makedirs(os.path.join(root_dataset, "images"))
        copyfile(img_path, os.path.join(root_dataset, "images", "{}.png".format(item)))
        
        np_img = np.array(Image.open(img_path))
        height, width = np_img.shape[:2]
        np_target_mask = np.zeros(shape=(height, width), dtype=np.uint8)
        for target_item in tqdm(os.listdir(os.path.join(root_dataset, "train_data", item, "masks"))):
            target_mask_path = os.path.join(root_dataset, "train_data", item, "masks", "{}".format(target_item))
            mask = Image.open(target_mask_path)
            np_mask = np.array(mask, dtype=np.uint8) / 255
            np_mask = np_mask.astype(np.uint8)
            np_target_mask += np_mask
        np_target_mask[np_target_mask > 0] = 1
        np_target_mask *= 255
        target_mask = Image.fromarray(np_target_mask).convert('RGB')
        if not os.path.exists(os.path.join(root_dataset, "targets")):
            os.makedirs(os.path.join(root_dataset, "targets"))
        target_mask.save(os.path.join(root_dataset, "targets", "{}.png".format(item)))


if __name__ == '__main__':
    main()