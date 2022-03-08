import torch

from hwad_bench.matchers import MeanEmbeddingMatcher, NearestMatcher


def test_mean_embdiing_matcher() -> None:
    matcher = MeanEmbeddingMatcher()
    embeddings = torch.tensor([[1, 2], [3, 4]]).float()
    labels = torch.tensor([0, 3]).long()
    matcher.update(embeddings, labels)
    matcher.update(embeddings, labels)
    index = matcher.create_index()
    assert index[0].shape == (2,)
    tgt_embeddings = torch.tensor([[1, 2], [3, 4], [-4, -3]]).float()
    values, matched = matcher(tgt_embeddings, k=2)
    assert matched.tolist() == [[0, 3], [3, 0], [0, 3]]


def test_nearest_embdiing_matcher() -> None:
    matcher = NearestMatcher()
    embeddings = torch.tensor([[1, 2], [3, 4], [2, 3], [-1, -2]]).float()
    labels = torch.tensor([0, 1, 2, 3]).long()
    matcher.update(embeddings, labels)
    matcher.create_index()
    tgt_embeddings = torch.tensor([[1, 2], [3, 4], [-4, -3]]).float()
    values, matched = matcher(tgt_embeddings, k=2)
    assert matched.tolist() == [[0, 2], [1, 2], [3, 0]]
