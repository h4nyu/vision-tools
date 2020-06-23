import torch
from object_detection.models.transformer import Transformer
from torch import nn


def test_multihead() -> None:
    batch_size = 10
    embed_dim = 10
    source_seqence = 7
    target_sequence = 9

    query = torch.rand(target_sequence, batch_size, embed_dim)
    key = torch.ones(source_seqence, batch_size, embed_dim)
    value = torch.rand(source_seqence, batch_size, embed_dim)
    key_padding_mask = torch.zeros((batch_size, source_seqence), dtype=torch.bool)

    fn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, dropout=0.1)
    target, memory = fn(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
        key_padding_mask=key_padding_mask,
    )
    assert target.shape == (target_sequence, batch_size, embed_dim)


def test_embed() -> None:
    hidden_dim = 10
    batch_size = 2
    num_features = 128
    num_classes = 2
    fn = nn.Linear(hidden_dim, num_classes + 1)
    hs = torch.rand((hidden_dim, num_features, batch_size))
    #  print(f"{hs.shape=}")
    #  res = fn(hs)


def test_transformer() -> None:
    batch_size = 2
    height = 34
    width = 34
    num_queries = 100
    hidden_dim = 256

    src = torch.rand((batch_size, hidden_dim, height, width))
    mask = torch.zeros((batch_size, height, width))
    query_embed = torch.rand((num_queries, hidden_dim))
    pos_embed = torch.rand((batch_size, hidden_dim, height, width))

    fn = Transformer(d_model=hidden_dim)

    res, _ = fn(src, mask, query_embed, pos_embed)

    assert res.shape == (6, batch_size, num_queries, hidden_dim)
