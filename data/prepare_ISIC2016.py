import os
from shutil import copyfile

from tqdm import tqdm


DATA_ROOT = "ISIC2016"


def main():
    import sys
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    root_dataset = os.path.join(script_directory, DATA_ROOT)

    for item in tqdm(os.listdir(os.path.join(root_dataset, 'targets'))):
        img_path = os.path.join(root_dataset, 'targets', item)
        os.rename(img_path, os.path.join(root_dataset, 'targets', "{}.jpg".format(item.rsplit("_", 1)[0])))


if __name__ == '__main__':
    main()