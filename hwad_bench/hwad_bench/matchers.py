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

    def __call__(self, embeddings: Tensor, k: int) -> tuple[Tensor, Tensor]:
        distance = self.distance(embeddings, self.index).nan_to_num(float("-inf"))
        return torch.topk(distance, k, dim=1)


class NearestMatcher:
    def __init__(self) -> None:
        self.embeddings_list: list[Tensor] = []
        self.labels_list: list[Tensor] = []
        self.embeddings: Tensor = torch.zeros(0, 0)
        self.labels: Tensor = torch.zeros(0)
        self.max_classes = 0
        self.embedding_size = 0
        self.distance = CosineSimilarity()

    def update(self, embeddings: Tensor, labels: Tensor) -> None:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        self.embeddings_list.append(embeddings)
        self.labels_list.append(labels)

    def create_index(self) -> None:
        self.embeddings = torch.cat(self.embeddings_list, dim=0)
        self.labels = torch.cat(self.labels_list, dim=0)

    def __call__(self, embeddings: Tensor, k: int) -> tuple[Tensor, Tensor]:
        distance = self.distance(embeddings, self.embeddings)
        matched_distance, indices = distance.topk(k, dim=1)
        matched_labels = self.labels[indices]
        return matched_distance, matched_labels
