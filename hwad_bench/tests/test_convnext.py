import torch
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore

from hwad_bench.convnext import ConvNeXt, MeanEmbeddingMmatcher


def test_model() -> None:
    embedding_size = 2
    model = ConvNeXt(name="convnext_tiny", embedding_size=embedding_size)
    images = torch.randn(2, 3, 256, 256)
    output = model(images)
    assert output.shape == (2, embedding_size)


def test_knn() -> None:
    matcher = MeanEmbeddingMmatcher()
    embeddings = torch.tensor([[1, 2], [3, 4]]).float()
    labels = torch.tensor([0, 3]).long()
    matcher.update(embeddings, labels)
    matcher.update(embeddings, labels)
    index = matcher.create_index()
    assert index[0].shape == (2,)
    tgt_embeddings = torch.tensor([[1, 2], [3, 4], [-4, -3]]).float()
    res = matcher(tgt_embeddings)
    values, matched = torch.topk(res, 2)
    assert matched.tolist() == [[0, 3], [3, 0], [0, 3]]


def test_scheduler() -> None:
    model = ConvNeXt(name="convnext_tiny", embedding_size=10)
    optimizer = optim.AdamW(model.parameters())
    scheduler = OneCycleLR(
        optimizer,
        total_steps=100,
        max_lr=1.0,
        pct_start=0.1,
    )
    for i in range(5, 100):
        scheduler.step(i)
        print(i, scheduler.get_last_lr())
