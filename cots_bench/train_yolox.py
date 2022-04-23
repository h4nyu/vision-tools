import os

from cots_bench.yolox import train
from vision_tools.utils import load_config

if __name__ == "__main__":
    cfg = load_config("./config/yolox.4.yaml")
    train(cfg)
