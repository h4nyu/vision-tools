from torch import optim
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore

from hwad_bench.convnext import ConvNeXt
from hwad_bench.scheduler import WarmupReduceLROnPlaetou


def test_scheduler() -> None:
    model = ConvNeXt(name="convnext_tiny", embedding_size=10)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    scheduler = WarmupReduceLROnPlaetou(
        optimizer,
        warmup_steps=10,
        max_lr=0.1,
        patience=1,
    )
    for i in range(11, 15):
        scheduler.step(0.1, i)
        print(i, scheduler.get_last_lr())
