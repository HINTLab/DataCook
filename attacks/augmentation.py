"""Strong stochastic augmentation used during unauthorized training."""

from __future__ import annotations

from typing import Any

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ._utils import as_nchw_tensor, restore_images


def strong_augment(images: Any, strength: int = 2, seed: int | None = None) -> Any:
    """Apply crop/flip/rotation/jitter/blur/RandAugment/erasing to a batch."""
    if strength < 1:
        raise ValueError("strength must be at least 1.")
    tensor, info = as_nchw_tensor(images)
    height, width = tensor.shape[-2:]
    channels = tensor.shape[1]
    operators = [
        transforms.RandomResizedCrop((height, width), scale=(max(0.5, 1.0 - 0.15 * strength), 1.0), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10 * strength, interpolation=InterpolationMode.BILINEAR),
    ]
    if channels == 3:
        operators.append(transforms.ColorJitter(0.1 * strength, 0.1 * strength, 0.1 * strength, 0.03 * strength))
    operators.append(transforms.GaussianBlur(kernel_size=3 if min(height, width) < 64 else 5, sigma=(0.1, 0.5 * strength)))
    pipeline = transforms.Compose(operators)
    rand_augment = transforms.RandAugment(
        num_ops=min(4, strength + 1), magnitude=min(30, 5 * strength)
    ) if hasattr(transforms, "RandAugment") else None
    erasing = transforms.RandomErasing(
        p=min(0.8, 0.25 * strength), scale=(0.01, min(0.2, 0.05 * strength))
    )
    def apply_augmentation() -> torch.Tensor:
        augmented_images = []
        for image in tensor:
            image = pipeline(image)
            # Older torchvision RandAugment expects uint8 for solarize/posterize.
            if rand_augment is not None:
                image = rand_augment((image.clamp(0, 1) * 255).round().to(torch.uint8)).float() / 255.0
            augmented_images.append(erasing(image))
        return torch.stack(augmented_images)

    if seed is None:
        # During robust fine-tuning, consume the caller's seeded RNG normally so
        # successive mini-batches receive different (but reproducible) views.
        augmented = apply_augmentation()
    else:
        devices = [tensor.device] if tensor.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            augmented = apply_augmentation()
    return restore_images(augmented, info)
