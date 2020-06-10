import torch
import typing as t
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from torch import Tensor


def plot_boxes(boxes: Tensor, path: str, size: t.Tuple[int, int] = (128, 128)) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    h, w = size
    ax.grid(False)
    ax.imshow(torch.ones(102, h, 3))
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * w
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * h
    for box in boxes:
        rect = mpatches.Rectangle(
            (box[0] - box[2] / 2, box[1] - box[3] / 2),
            width=box[2],
            height=box[3],
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)
    plt.savefig(path)
    plt.close()


def plot_heatmap(heatmap: Tensor, path: t.Union[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    img = heatmap.permute(1, 2, 0)
    #  h, w = heatmap.shape
    ax.grid(False)
    ax.imshow(img)
    plt.savefig(path)
    plt.close()
