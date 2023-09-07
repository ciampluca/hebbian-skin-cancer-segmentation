import os
from shutil import copyfile

from tqdm import tqdm


SOURCE_ROOT = "PH2Dataset/PH2 Dataset images"
TARGET_ROOT = "PH2"


def main():
    import sys
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    source_dataset = os.path.join(script_directory, SOURCE_ROOT)
    target_dataset = os.path.join(script_directory, TARGET_ROOT)

    if not os.path.exists(target_dataset):
        os.makedirs(target_dataset)

    for item in tqdm(os.listdir(source_dataset)):
        img_path = os.path.join(source_dataset, item, "{}_Dermoscopic_Image".format(item), "{}.bmp".format(item))
        if not os.path.exists(os.path.join(target_dataset, "images")):
            os.makedirs(os.path.join(target_dataset, "images"))
        copyfile(img_path, os.path.join(target_dataset, "images", "{}.bmp".format(item)))
        
        target_path = os.path.join(source_dataset, item, "{}_lesion".format(item), "{}_lesion.bmp".format(item))
        if not os.path.exists(os.path.join(target_dataset, "targets")):
            os.makedirs(os.path.join(target_dataset, "targets"))
        copyfile(target_path, os.path.join(target_dataset, "targets", "{}.bmp".format(item)))


if __name__ == '__main__':
    main()