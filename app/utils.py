import torch
import typing as t
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from torch import Tensor


def plot_boxes(
    path: str,
    boxes: Tensor,
    probs: t.Optional[Tensor] = None,
    size: t.Tuple[int, int] = (128, 128),
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    w, h = size
    ax.grid(False)
    ax.imshow(torch.ones(102, h, 3))
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * w
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * h
    _probs = probs if probs is not None else torch.ones((w, h))
    for box, p in zip(boxes, _probs):
        ax.text(box[0], box[1], f"{p:.2f}", fontsize=5)
        rect = mpatches.Rectangle(
            (box[0] - box[2] / 2, box[1] - box[3] / 2),
            width=box[2],
            height=box[3],
            fill=False,
            edgecolor="red",
            linewidth=2,
            alpha=float(p),
        )
        ax.add_patch(rect)
    plt.savefig(path)
    plt.close()


def plot_heatmap(heatmap: Tensor, path: t.Union[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(False)
    ax.imshow(heatmap, interpolation="nearest")
    plt.savefig(path)
    plt.close()
