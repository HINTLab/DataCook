from __future__ import annotations

from typing import Any

import numpy as np
import torch


def as_nchw_tensor(images: Any) -> tuple[torch.Tensor, tuple[str, Any]]:
    """Convert supported image batches to NCHW while retaining restoration info."""
    is_tensor = torch.is_tensor(images)
    original_device = images.device if is_tensor else None
    original_dtype = images.dtype if is_tensor else np.asarray(images).dtype
    tensor = images.detach().clone() if is_tensor else torch.as_tensor(np.asarray(images))
    layout = "nchw"
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
        layout = "nhw"
    elif tensor.ndim == 4 and tensor.shape[1] not in (1, 3) and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
        layout = "nhwc"
    elif tensor.ndim != 4 or tensor.shape[1] not in (1, 3):
        raise ValueError("Expected image batch with grayscale or RGB channels.")
    scale = 255.0 if not torch.is_floating_point(tensor) or float(tensor.max()) > 1.5 else 1.0
    tensor = tensor.float() / scale
    return tensor, ("tensor" if is_tensor else "numpy", original_device, original_dtype, layout, scale)


def restore_images(tensor: torch.Tensor, info: tuple[str, Any]) -> Any:
    source, device, dtype, layout, scale = info
    tensor = tensor.clamp(0, 1) * scale
    if layout == "nhw":
        tensor = tensor[:, 0]
    elif layout == "nhwc":
        tensor = tensor.permute(0, 2, 3, 1)
    if source == "tensor":
        if not torch.empty((), dtype=dtype).is_floating_point():
            tensor = tensor.round()
        return tensor.to(device=device, dtype=dtype)
    array = tensor.cpu().numpy()
    if np.issubdtype(dtype, np.integer):
        array = np.rint(array)
    return array.astype(dtype)
