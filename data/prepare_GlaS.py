import os
from shutil import copyfile

from tqdm import tqdm


SOURCE_ROOT = "Warwick_QU_Dataset"
TARGET_ROOT = "GlaS"


def main():
    import sys
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    source_dataset = os.path.join(script_directory, SOURCE_ROOT)
    target_dataset = os.path.join(script_directory, TARGET_ROOT)

    if not os.path.exists(target_dataset):
        os.makedirs(target_dataset)

    img_names, target_names = [], []
    for item in os.listdir(source_dataset):
        if item.endswith("_anno.bmp"):
            target_names.append(item)
        elif item.endswith(".csv"):
            continue
        else:
            img_names.append(item)

    for item in tqdm(img_names):
        if item.startswith("train"):
            if not os.path.exists(os.path.join(target_dataset, "train", "images")):
                os.makedirs(os.path.join(target_dataset, "train", "images"))
            img_path = os.path.join(source_dataset, item)
            copyfile(img_path, os.path.join(target_dataset, "train", "images", item))
            if not os.path.exists(os.path.join(target_dataset, "train", "targets")):
                os.makedirs(os.path.join(target_dataset, "train", "targets"))
            target_path = os.path.join(source_dataset, item.rsplit(".bmp", 1)[0] + "_anno.bmp")
            copyfile(target_path, os.path.join(target_dataset, "train", "targets", item))
        else:
            if not os.path.exists(os.path.join(target_dataset, "test", "images")):
                os.makedirs(os.path.join(target_dataset, "test", "images"))
            img_path = os.path.join(source_dataset, item)
            copyfile(img_path, os.path.join(target_dataset, "test", "images", item))
            if not os.path.exists(os.path.join(target_dataset, "test", "targets")):
                os.makedirs(os.path.join(target_dataset, "test", "targets"))
            target_path = os.path.join(source_dataset, item.rsplit(".bmp", 1)[0] + "_anno.bmp")
            copyfile(target_path, os.path.join(target_dataset, "test", "targets", item))
        

if __name__ == '__main__':
    main()