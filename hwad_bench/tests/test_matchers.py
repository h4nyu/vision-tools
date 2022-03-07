import torch

from hwad_bench.matchers import MeanEmbeddingMatcher


def test_mean_embdiing_matcher() -> None:
    matcher = MeanEmbeddingMatcher()
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
