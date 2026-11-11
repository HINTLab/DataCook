"""Small raw-subset fine-tuning attack for PyTorch classifiers."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader


def fine_tune_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    *,
    task_type: str,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    device: str | torch.device | None = None,
    input_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    seed: int = 0,
) -> torch.nn.Module:
    """Return a fine-tuned copy; the input model and checkpoint stay untouched."""
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive.")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    attacked_model = copy.deepcopy(model).to(selected_device)
    optimizer = torch.optim.Adam(attacked_model.parameters(), lr=learning_rate)
    normalized_task = task_type.replace("-", "").replace("_", "").lower()
    if normalized_task not in {"binary", "multiclass", "multilabel"}:
        raise ValueError(f"Unsupported task_type {task_type!r}.")
    if normalized_task in {"multilabel", "binary"}:
        binary_criterion: Any = torch.nn.BCEWithLogitsLoss()
    if normalized_task in {"binary", "multiclass"}:
        class_criterion: Any = torch.nn.CrossEntropyLoss()

    devices = [selected_device] if selected_device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        attacked_model.train()
        for _ in range(epochs):
            for batch in data_loader:
                if len(batch) < 2:
                    raise ValueError("Fine-tuning loader batches must contain (inputs, labels).")
                inputs, labels = batch[0].to(selected_device), batch[1].to(selected_device)
                if input_transform is not None:
                    inputs = input_transform(inputs)
                optimizer.zero_grad(set_to_none=True)
                logits = attacked_model(inputs)
                if normalized_task == "multilabel":
                    loss = binary_criterion(logits, labels.float())
                elif normalized_task == "binary" and (logits.ndim == 1 or logits.shape[-1] == 1):
                    loss = binary_criterion(logits.reshape(-1), labels.reshape(-1).float())
                else:
                    loss = class_criterion(logits, labels.reshape(-1).long())
                loss.backward()
                optimizer.step()
    return attacked_model.eval()
