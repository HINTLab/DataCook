"""Simple, dependency-free median-filter purification baseline."""

from __future__ import annotations

from typing import Any

import torch.nn.functional as functional

from ._utils import as_nchw_tensor, restore_images


def median_filter(images: Any, kernel_size: int = 3) -> Any:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    tensor, info = as_nchw_tensor(images)
    padding = kernel_size // 2
    padded = functional.pad(tensor, [padding] * 4, mode="reflect")
    patches = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    filtered = patches.contiguous().view(*patches.shape[:4], -1).median(dim=-1).values
    return restore_images(filtered, info)
