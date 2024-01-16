import os
from tqdm import tqdm


ROOT = "/home/luca/dgx-a100/workspace/hebbian-skin-cancer-segmentation/runs"
DATASET = "ph2"

METRICS_TO_REMOVE = ['95hd', 'asd']


def main():
    exp_path = "experiment={}".format(DATASET)
    root_path = os.path.join(ROOT, exp_path)

    for exp in tqdm(os.listdir(root_path)):
        for inv_temp in os.listdir(os.path.join(root_path, exp)):
            for regime in os.listdir(os.path.join(root_path, exp, inv_temp)):
                for run in os.listdir(os.path.join(root_path, exp, inv_temp, regime)):
                    snapshot_path = os.path.join(root_path, exp, inv_temp, regime, run, 'best_models')
                    for snapshot in os.listdir(snapshot_path):
                        metric_name = snapshot.rsplit("-", 1)[1].split(".", 1)[0]
                        if metric_name in METRICS_TO_REMOVE:
                            snapshot_to_remove_path = os.path.join(snapshot_path, snapshot)
                            os.remove(snapshot_to_remove_path)


if __name__ == "__main__":
    main()