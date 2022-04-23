import sys

from vision_tools.utils import load_config
from vision_tools.yolox import YOLOX, Criterion, Inference

sys.path.append("/kaggle/input/tensorflow-great-barrier-reef")
import os

import greatbarrierreef
from tqdm import tqdm

env = greatbarrierreef.make_env()


def predict() -> None:
    cfg = load_config(os.path.join(os.path.dirname(__file__), "config/yolox.yaml"))
    iter_test = env.iter_test()
    for idx, (img, pred_df) in enumerate(tqdm(iter_test)):
        anno = ""


if __name__ == "__main__":
    predict()
