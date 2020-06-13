import torch
import typing as t
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from torch import Tensor


class DetectionPlot:
    def __init__(self, size: t.Tuple[int, int] = (128, 128)) -> None:
        self.w, self.h = size
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.imshow(torch.ones(self.w, self.h, 3),interpolation="nearest")

    def __del__(self) -> None:
        plt.close(self.fig)

    def save(self, path: t.Union[str, Path]) -> None:
        self.fig.savefig(path)

    def with_image(self, image: Tensor) -> None:
        if len(image.shape) == 2:
            self.ax.imshow(image, interpolation="nearest")
            self.w, self.h = image.shape
        elif len(image.shape) == 3:
            image = image.permute(1, 2, 0)
            self.ax.imshow(image, interpolation="nearest")
            self.w, self.h, _ = image.shape
        else:
            shape = image.shape
            raise ValueError(f"invald {shape=}")

    def with_boxes(
        self,
        boxes: Tensor,
        probs: t.Optional[Tensor] = None,
        color: str = "black",
        fontsize: int = 7,
    ) -> None:
        b, _ = boxes.shape
        _probs = probs if probs is not None else torch.ones((b,))
        _boxes = boxes.clone()
        _boxes[:, [0, 2]] = boxes[:, [0, 2]] * self.w
        _boxes[:, [1, 3]] = boxes[:, [1, 3]] * self.h
        for box, p in zip(_boxes, _probs):
            x0 = box[0] - box[2] / 2
            y0 = box[1] - box[3] / 2
            self.ax.text(x0, y0, f"{p:.2f}", fontsize=fontsize, color=color)
            rect = mpatches.Rectangle(
                (x0, y0),
                width=box[2],
                height=box[3],
                fill=False,
                edgecolor=color,
                linewidth=2,
                alpha=float(p),
            )
            self.ax.add_patch(rect)


def plot_boxes(
    path: str,
    boxes: Tensor,
    probs: t.Optional[Tensor] = None,
    size: t.Tuple[int, int] = (128, 128),
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    w, h = size
    ax.grid(False)
    ax.imshow(torch.ones(w, h, 3))
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
