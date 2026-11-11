"""JPEG compression purification attack."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image

from ._utils import as_nchw_tensor, restore_images


def jpeg_compress(images: Any, quality: int = 75) -> Any:
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100].")
    tensor, info = as_nchw_tensor(images)
    output = []
    for image in tensor.cpu():
        uint8 = (image.clamp(0, 1) * 255).round().byte()
        if uint8.shape[0] == 1:
            pil = Image.fromarray(uint8[0].numpy(), mode="L")
        else:
            pil = Image.fromarray(uint8.permute(1, 2, 0).numpy(), mode="RGB")
        buffer = BytesIO()
        pil.save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        decoded = np.asarray(Image.open(buffer)).copy()
        if decoded.ndim == 2:
            decoded = decoded[None]
        else:
            decoded = np.transpose(decoded, (2, 0, 1))
        output.append(torch.from_numpy(decoded).float() / 255.0)
    return restore_images(torch.stack(output).to(tensor.device), info)
