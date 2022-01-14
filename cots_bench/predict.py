import sys
from omegaconf import OmegaConf
from vision_tools.yolox import YOLOX, Criterion, Inference

sys.path.append("/kaggle/input/tensorflow-great-barrier-reef")
import os
import greatbarrierreef
from tqdm import tqdm

env = greatbarrierreef.make_env()


def predict() -> None:
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config/yolox.yaml"))
    iter_test = env.iter_test()
    for idx, (img, pred_df) in enumerate(tqdm(iter_test)):
        anno = ""
        print(pred_df)
        # print(img.shape)


if __name__ == "__main__":
    predict()
