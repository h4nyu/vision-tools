import pytest
import torch
from torch import Tensor
import random
from typing import Tuple, List, Any
from torchvision.ops import box_convert
from cots_bench.metric import BoxF2
from vision_tools.utils import batch_draw, draw, load_config
from torch.utils.tensorboard import SummaryWriter
from cots_bench.data import (
    COTSDataset,
    read_train_rows,
    Transform,
    Row,
)
from toolz.curried import pipe, filter

cfg = load_config("/app/cots_bench/config/yolox.yaml")
writer = SummaryWriter("runs/test")


@pytest.fixture
def rows() -> List[Row]:
    return read_train_rows(cfg["dataset_dir"])


@pytest.fixture
def dataset(rows: List[Row]) -> COTSDataset:
    return COTSDataset(rows, transform=Transform(cfg))


def generate_pred(gt_boxes: Tensor) -> Tuple[Tensor, Tensor]:
    confs = []
    gt_cxcys = box_convert(gt_boxes, in_fmt="xyxy", out_fmt="cxcywh")
    pred_cxcys: List[Tensor] = []
    for gt_cxcy in gt_cxcys:
        # pseudo pred bbox
        conf = random.random()
        confs.append(conf)
        #         noise = (np.random.randn(4)*5).round()
        pred_cxcy = gt_cxcy * torch.tensor([1.0, 1.0, 1.25, 1.25])
        pred_cxcys.append(pred_cxcy)

    # print(torch.cat(pred_cxcys, dim=1).shape)
    pred_boxes = box_convert(torch.stack(pred_cxcys), in_fmt="cxcywh", out_fmt="xyxy")
    pred_confs = torch.tensor(confs)
    return pred_confs, pred_boxes


def test_boxf2(dataset: COTSDataset) -> None:
    sample = dataset[4515]
    image = sample["image"]
    gt_boxes = sample["boxes"]
    plot = draw(image=image, boxes=gt_boxes)
    writer.add_image("metric", plot, 0)
    pred_scores, pred_boxes = generate_pred(gt_boxes)
    plot = draw(image=image, boxes=pred_boxes)
    writer.add_image("metric", plot, 1)
    writer.flush()

    metric = BoxF2(iou_thresholds=[0.5])
    gt_batch: Any = {
        "box_batch": [gt_boxes],
    }
    pred_batch: Any = {
        "box_batch": [pred_boxes],
    }
    metric.accumulate(pred_batch, gt_batch)
    assert metric.value[0] == pytest.approx(1.0)
