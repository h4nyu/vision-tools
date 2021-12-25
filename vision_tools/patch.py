import torch.nn as nn
import torch
from torch import Tensor
from typing import Callable, Optional


@torch.no_grad()
def grid(h: int, w: int, dtype: Optional[torch.dtype] = None) -> tuple[Tensor, Tensor]:
    grid_y, grid_x = torch.meshgrid(  # type:ignore
        torch.arange(h, dtype=dtype),
        torch.arange(w, dtype=dtype),
    )
    return (grid_y, grid_x)


class ToPatches:
    def __init__(
        self,
        patch_size: int,
        use_reflect: bool = False,
    ) -> None:
        self.patch_size = patch_size
        self.use_reflect = use_reflect

    def __call__(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        device = images.device
        b, c, h, w = images.shape
        pad_size = (
            0,
            (w // self.patch_size + 1) * self.patch_size - w,
            0,
            (h // self.patch_size + 1) * self.patch_size - h,
        )
        if self.use_reflect:
            pad: Callable[[Tensor], Tensor] = nn.ReflectionPad2d(pad_size)

        else:
            pad = nn.ZeroPad2d(pad_size)
        images = pad(images)
        _, _, padded_h, padded_w = images.shape
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = (
            patches.permute([0, 2, 3, 1, 4, 5])
            .contiguous()
            .view(b, -1, 3, self.patch_size, self.patch_size)
        )
        index = torch.stack(
            grid(padded_w // self.patch_size, padded_h // self.patch_size)
        ).to(device)
        patch_grid = index.permute([2, 1, 0]).contiguous().view(-1, 2) * self.patch_size
        return images, patches, patch_grid


class MergePatchedMasks:
    def __init__(
        self,
        patch_size: int,
    ) -> None:
        self.patch_size = patch_size

    def __call__(self, mask_batch: list[Tensor], patch_grid: Tensor) -> Tensor:
        device = patch_grid.device
        last_grid = patch_grid[-1]
        out_size = last_grid + self.patch_size
        out_batch: list[Tensor] = []
        for masks, grid in zip(mask_batch, patch_grid):
            out_masks = torch.zeros(
                (len(masks), int(out_size[1]), int(out_size[0])), dtype=masks.dtype
            ).to(device)
            out_masks[
                :,
                grid[1] : grid[1] + self.patch_size,
                grid[0] : grid[0] + self.patch_size,
            ] = masks
            out_batch.append(out_masks)
        return torch.cat(out_batch)
