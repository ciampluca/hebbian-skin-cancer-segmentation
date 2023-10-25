import os
from shutil import copyfile

from tqdm import tqdm


SOURCE_ROOT = "Kvasir-SEG"
TARGET_ROOT = "KvasirSEG"


def main():
    import sys
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    source_dataset = os.path.join(script_directory, SOURCE_ROOT)
    target_dataset = os.path.join(script_directory, TARGET_ROOT)

    if not os.path.exists(target_dataset):
        os.makedirs(target_dataset)

    for item in tqdm(os.listdir(os.path.join(source_dataset, 'images'))):
        img_path = os.path.join(os.path.join(source_dataset, 'images'), item)
        if not os.path.exists(os.path.join(target_dataset, "images")):
            os.makedirs(os.path.join(target_dataset, "images"))
        copyfile(img_path, os.path.join(target_dataset, "images", item))
        
        target_path = img_path.replace("/images/", "/masks/")
        if not os.path.exists(os.path.join(target_dataset, "targets")):
            os.makedirs(os.path.join(target_dataset, "targets"))
        copyfile(target_path, os.path.join(target_dataset, "targets", item))


if __name__ == '__main__':
    main()