import os
from shutil import copyfile

from tqdm import tqdm


DATA_ROOT = "DRIVE"


def main():
    import sys
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    root_dataset = os.path.join(script_directory, DATA_ROOT)

    for item in tqdm(os.listdir(os.path.join(root_dataset, 'images'))):
        img_path = os.path.join(root_dataset, 'images', item)
        os.rename(img_path, os.path.join(root_dataset, 'images', "{}.tif".format(item.split("_", 1)[0])))

    for item in tqdm(os.listdir(os.path.join(root_dataset, 'targets'))):
        img_path = os.path.join(root_dataset, 'targets', item)
        os.rename(img_path, os.path.join(root_dataset, 'targets', "{}.tif".format(item.split("_", 1)[0])))


if __name__ == '__main__':
    main()