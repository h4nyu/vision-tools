from cots_bench.yolox import evaluate
import os
from vision_tools.utils import load_config

if __name__ == "__main__":
    config_file = "./config/ensemble.yaml"
    evaluate(config_file)
