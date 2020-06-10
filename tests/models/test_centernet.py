import torch
from app.dataset import Targets
from app.models.centernet import (
    #  BoxRegression,
    #  LabelClassification,
    Backbone,
    CenterNet,
    CenterHeatMap,
)
from app.utils import plot_heatmap


#  def test_boxregression() -> None:
#      inputs = torch.rand((10, 128, 32, 32))
#      fn = BoxRegression(in_channels=128, num_queries=100)
#      outs = fn(inputs)
#      assert outs.shape == (10, 100, 4)
#
#
#  def test_labelclassification() -> None:
#      inputs = torch.rand((10, 128, 32, 32))
#      fn = LabelClassification(in_channels=128, num_queries=100, num_classes=2)
#      outs = fn(inputs)
#      assert outs.shape == (10, 100, 2)




def test_heatmap() -> None:
    target = {
        "labels": torch.tensor([1, 0]).long(),
        "boxes": torch.tensor([[0.1, 0.1, 0.9, 0.1], [0.3, 0.4, 0.1, 0.1]]).float(),
    }
    fn = CenterHeatMap(w=100, h=100, kernel_size=5)
    res = fn(target)[0]
    c, _, _ = res.shape
    plot_heatmap(res, f"/store/plot/test-heatmap.png")

    #  for i in range(c):
    #      plot_heatmap(res[i], f"/store/plot/test-heatmap-{i}.png")


def test_backbone() -> None:
    inputs = torch.rand((10, 3, 1024, 1024))
    fn = Backbone("resnet34", out_channels=128)
    outs = fn(inputs)
    for o in outs:
        assert o.shape[1] == 128


def test_centernet() -> None:
    inputs = torch.rand((1, 3, 1024, 1024))
    num_classes = 1
    fn = CenterNet(num_classes=num_classes)
    outc, outr = fn(inputs)
    assert outc.shape == (1, num_classes, 1024 / 2, 1024 / 2)
