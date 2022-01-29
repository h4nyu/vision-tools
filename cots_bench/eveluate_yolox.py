from cots_bench.yolox import evaluate
import os
from vision_tools.utils import load_config

if __name__ == "__main__":
    config_file = os.getenv("CONFIG_FILE", "config.yaml")
    cfg = load_config(os.path.join("./config", config_file))
    evaluate(cfg)
