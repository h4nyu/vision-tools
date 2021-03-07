import torchvision, os
from bench.kuzushiji.data import read_rows, KuzushijiDataset, inv_normalize
from vnet.utils import DetectionPlot
from bench.kuzushiji import config


def test_read_rows() -> None:
    rows = read_rows(config.root_dir)
    assert len(rows) == 3605


def test_dataset() -> None:
    rows = read_rows(config.root_dir)
    dataset = KuzushijiDataset(rows)
    sample = dataset[0]
    id, img, boxes, labels = sample
    plot = DetectionPlot(inv_normalize(img))
    plot.draw_boxes(boxes)
    plot.save(os.path.join(config.root_dir, "test_dataset.png"))
