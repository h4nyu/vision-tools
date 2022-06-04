import torch

from tanacho_bench import Net


def test_net() -> None:
    net = Net(
        embedding_size=2,
        name="tf_efficientnet_b3_ns",
    )
    images = torch.randn(2, 3, 256, 256)
    out = net(images)
    print(out.shape)
