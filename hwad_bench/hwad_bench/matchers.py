from __future__ import annotations

import torch
from pytorch_metric_learning.distances import CosineSimilarity
from torch import Tensor
from torch.nn import functional as F


class MeanEmbeddingMatcher:
    def __init__(self) -> None:
        self.embeddings: dict[int, list[Tensor]] = {}
        self.index: Tensor = torch.zeros(0, 0)
        self.max_classes = 0
        self.embedding_size = 0
        self.distance = CosineSimilarity()

    def update(self, embeddings: Tensor, labels: Tensor) -> None:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        for label, embedding in zip(labels, embeddings):
            label = int(label)
            old_embedding = self.embeddings.get(label, [])
            self.embeddings[label] = old_embedding + [embedding]
            if self.max_classes < label:
                self.max_classes = label
            self.embedding_size = embedding.shape[-1]

    def create_index(self) -> Tensor:
        index = torch.full((self.max_classes + 1, self.embedding_size), float("nan"))
        if len(list(self.embeddings.values())) > 0:
            device = list(self.embeddings.values())[0][0]
            index = index.to(device)

        for label, embeddings in self.embeddings.items():
            index[int(label)] = torch.mean(torch.stack(embeddings), dim=0)
        self.index = index
        return index

    def __call__(self, embeddings: Tensor) -> Tensor:
        return self.distance(embeddings, self.index).nan_to_num(float("-inf"))


class NearestMatcher:
    def __init__(self) -> None:
        self.embeddings: dict[int, list[Tensor]] = {}
        self.index: Tensor = torch.zeros(0, 0)
        self.max_classes = 0
        self.embedding_size = 0
        self.distance = CosineSimilarity()

    def update(self, embeddings: Tensor, labels: Tensor) -> None:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        for label, embedding in zip(labels, embeddings):
            label = int(label)
            old_embedding = self.embeddings.get(label, [])
            self.embeddings[label] = old_embedding + [embedding]
            if self.max_classes < label:
                self.max_classes = label
            self.embedding_size = embedding.shape[-1]

    def create_index(self) -> Tensor:
        index = torch.full((self.max_classes + 1, self.embedding_size), float("nan"))
        if len(list(self.embeddings.values())) > 0:
            device = list(self.embeddings.values())[0][0]
            index = index.to(device)

        for label, embeddings in self.embeddings.items():
            index[int(label)] = torch.mean(torch.stack(embeddings), dim=0)
        self.index = index
        return index

    def __call__(self, embeddings: Tensor) -> Tensor:
        return self.distance(embeddings, self.index).nan_to_num(float("-inf"))
