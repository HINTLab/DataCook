"""I/O helpers for channel-first cooked medical volumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_cooked_array_npz(path: str | Path, images, labels) -> Path:
    """Atomically save normalized channel-first cooked arrays and labels.

    Unlike the legacy 3D helper, this function preserves a multi-dimensional
    label matrix such as ChestMNIST's ``(N,14)`` multi-hot targets.
    """
    output = Path(path)
    arrays = np.asarray(images, dtype=np.float32)
    targets = np.asarray(labels)
    if arrays.ndim not in {4, 5}:
        raise ValueError(
            f"Cooked images must have shape (N,C,H,W) or (N,C,D,H,W); got {arrays.shape}."
        )
    if targets.ndim not in {1, 2}:
        raise ValueError(f"Cooked labels must have shape (N,) or (N,L); got {targets.shape}.")
    if len(arrays) == 0 or len(arrays) != len(targets):
        raise ValueError(
            f"Cooked images/labels must be non-empty and aligned; "
            f"got {len(arrays)} images and {len(targets)} labels."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        perturb_images=arrays,
        perturb_labels=targets,
    )
    temporary.replace(output)
    return output


def save_cooked_volume_npz(path: str | Path, images, labels) -> Path:
    """Atomically save cooked volumes for ``Data3DFolderWithLabel``.

    Images use the ``(N, C, D, H, W)`` contract and labels are flattened to
    ``(N,)`` so CrossEntropyLoss receives one class index per volume.
    """
    output = Path(path)
    volumes = np.asarray(images, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    if volumes.ndim != 5:
        raise ValueError(
            f"Cooked 3D images must have shape (N,C,D,H,W); got {volumes.shape}."
        )
    if len(volumes) == 0 or len(volumes) != len(targets):
        raise ValueError(
            f"Cooked 3D images/labels must be non-empty and aligned; "
            f"got {len(volumes)} images and {len(targets)} labels."
        )
    return save_cooked_array_npz(output, volumes, targets)
