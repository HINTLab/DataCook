"""Gaussian smoothing attack implemented without optional SciPy dependencies."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

from ._utils import as_nchw_tensor, restore_images


def gaussian_smooth(images: Any, sigma: float = 1.0, kernel_size: int = 5) -> Any:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    tensor, info = as_nchw_tensor(images)
    coordinates = torch.arange(kernel_size, device=tensor.device, dtype=tensor.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(tensor.shape[1], 1, kernel_size, kernel_size)
    padded = functional.pad(tensor, [kernel_size // 2] * 4, mode="reflect")
    smoothed = functional.conv2d(padded, kernel, groups=tensor.shape[1])
    return restore_images(smoothed, info)
