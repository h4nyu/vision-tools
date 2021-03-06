#  import typing as t
#  import pytest
#  import numpy as np
#  import torch
#  from vnet.models.backbones import (
#      ResNetBackbone,
#      EfficientNetBackbone,
#      ModelName,
#  )
#
#
#  @pytest.mark.parametrize("name", [
#      ("resnet18"),
#      ("resnet34"),
#      ("resnet50"),
#  ])
#  def test_resnetbackbone(name: ModelName) -> None:
#      inputs = torch.rand((10, 3, 128, 128))
#      fn = ResNetBackbone(name, out_channels=128)
#      outs = fn(inputs)
#      for o in outs:
#          assert o.shape[1] == 128
#
#
#  def test_effnetbackbone() -> None:
#      inputs = torch.rand((10, 3, 512, 512))
#      fn = EfficientNetBackbone(1, out_channels=128)
#      outs = fn(inputs)
#      for o in outs:
#          assert o.shape[1] == 128
