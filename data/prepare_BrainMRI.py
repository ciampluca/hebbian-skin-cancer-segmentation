import os
from shutil import copyfile

from tqdm import tqdm


SOURCE_ROOT = "kaggle_3m"
TARGET_ROOT = "BrainMRI"


def main():
    import sys
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    source_dataset = os.path.join(script_directory, SOURCE_ROOT)
    target_dataset = os.path.join(script_directory, TARGET_ROOT)

    if not os.path.exists(target_dataset):
        os.makedirs(target_dataset)

    for item in tqdm(os.listdir(source_dataset)):
        if os.path.isdir(os.path.join(source_dataset, item)):
            for img in os.listdir(os.path.join(source_dataset, item)):
                if 'mask' not in img:
                    img_path = os.path.join(source_dataset, item, img)
                    if not os.path.exists(os.path.join(target_dataset, "images")):
                        os.makedirs(os.path.join(target_dataset, "images"))
                    copyfile(img_path, os.path.join(target_dataset, "images", img))
        
                    target_path = os.path.join(source_dataset, item, "{}_mask.tif".format(img.rsplit(".", 1)[0]))
                    if not os.path.exists(os.path.join(target_dataset, "targets")):
                        os.makedirs(os.path.join(target_dataset, "targets"))
                    copyfile(target_path, os.path.join(target_dataset, "targets", img))


if __name__ == '__main__':
    main()