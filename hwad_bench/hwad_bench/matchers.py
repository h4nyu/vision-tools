from __future__ import annotations

import torch
from pytorch_metric_learning.distances import CosineSimilarity
from torch import Tensor, nn
from torch.nn import functional as F


class MeanEmbeddingMatcher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
        self.embeddings = {}
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
        all_distance = self.distance(embeddings, self.embeddings)
        all_matched_distance, indices = all_distance.sort(
            dim=1,
            descending=True,
        )
        all_matched_labels = self.labels[indices]
        matched_distance = torch.full((embeddings.shape[0], k), float("nan"))
        matched_labels = torch.full((embeddings.shape[0], k), float("nan"))
        for i, (distance, labels) in enumerate(
            zip(all_matched_distance, all_matched_labels)
        ):
            topk_distance, topk_label = [], []
            unique_labels: set[int] = set()
            for distance, label in zip(distance, labels):
                if len(unique_labels) >= k:
                    break
                if label not in unique_labels:
                    topk_distance.append(float(distance))
                    topk_label.append(int(label))
                    unique_labels.add(label)
            left = k - len(topk_distance)
            if left > 0:
                topk_distance += [topk_distance[-1]] * left
                topk_label += [topk_label[-1]] * left
            matched_distance[i] = torch.tensor(topk_distance)
            matched_labels[i] = torch.tensor(topk_label)
        return matched_distance.to(embeddings), matched_labels.to(self.labels)
