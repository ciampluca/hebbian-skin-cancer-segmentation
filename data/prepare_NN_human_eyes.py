import os
from shutil import copyfile

import skimage
from tqdm import tqdm

import cv2
import numpy as np


SOURCE_ROOT = "NN_human_eyes"
TARGET_ROOT = "NN"


def main():
    import sys
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    source_dataset = os.path.join(script_directory, SOURCE_ROOT)
    target_dataset = os.path.join(script_directory, TARGET_ROOT)

    if not os.path.exists(target_dataset):
        os.makedirs(target_dataset)

    for item in tqdm(os.listdir(os.path.join(source_dataset, 'images'))):
        img_path = os.path.join(source_dataset, "images", item)
        if not os.path.exists(os.path.join(target_dataset, "images")):
            os.makedirs(os.path.join(target_dataset, "images"))
        image = skimage.io.imread(img_path)
        image = skimage.color.gray2rgb(image)
        skimage.io.imsave(os.path.join(target_dataset, 'images', '{}.png'.format(item.rsplit(".", 1)[0])), image)
        
        target_path = os.path.join(source_dataset, "targets", "{}.png".format(item.rsplit(".", 1)[0]))
        if not os.path.exists(os.path.join(target_dataset, "targets")):
            os.makedirs(os.path.join(target_dataset, "targets"))
        copyfile(target_path, os.path.join(target_dataset, "targets", "{}.png".format(item.rsplit(".", 1)[0])))


if __name__ == '__main__':
    main()