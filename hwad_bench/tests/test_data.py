from hwad_bench.data import read_annotations
from vision_tools.utils import load_config

dataset_cfg = load_config("config/dataset.yaml")["dataset"]


def test_read_annotations() -> None:
    """
    Test the read_annotations function.
    """
    annotations = read_annotations(dataset_cfg["train_annotation_path"])
    assert len(annotations) == 51033
    print(annotations[0])
