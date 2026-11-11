"""Adapters for loading DataCook/PyTorch models and producing probabilities."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .common import load_array, resolve_path


def import_object(dotted_path: str) -> Any:
    module_name, separator, attribute = dotted_path.rpartition(".")
    if not separator:
        raise ValueError(f"Expected a dotted import path, got {dotted_path!r}.")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def load_model(spec: Mapping[str, Any], base_dir: Path, device: str | torch.device) -> torch.nn.Module:
    """Load a model through a custom factory or an existing mlconfig YAML."""
    if "factory" in spec:
        factory = import_object(str(spec["factory"]))
        model = factory(**dict(spec.get("kwargs", {})))
    elif "config_path" in spec:
        import dataset  # noqa: F401 - registers mlconfig dataset
        import models  # noqa: F401 - registers mlconfig models
        import mlconfig

        config_path = resolve_path(str(spec["config_path"]), base_dir)
        config = mlconfig.load(str(config_path))
        model = config.model()
    else:
        raise ValueError("Model spec requires `factory` or `config_path`.")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Model factory did not return torch.nn.Module.")

    checkpoint_text = spec.get("checkpoint")
    if checkpoint_text:
        checkpoint_path = resolve_path(str(checkpoint_text), base_dir)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before weights_only was introduced
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, Mapping):
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        else:
            state = checkpoint
        if not isinstance(state, Mapping):
            raise ValueError(f"Could not find a state dict in {checkpoint_path}.")
        # Transparently handle checkpoints saved from DataParallel.
        if state and all(str(key).startswith("module.") for key in state):
            state = {str(key)[7:]: value for key, value in state.items()}
        model.load_state_dict(state, strict=bool(spec.get("strict", True)))
    return model.to(device).eval()


def probabilities_from_logits(logits: torch.Tensor, task_type: str) -> torch.Tensor:
    task = task_type.replace("-", "").replace("_", "").lower()
    if task == "multilabel":
        return torch.sigmoid(logits)
    if task == "binary" and (logits.ndim == 1 or logits.shape[-1] == 1):
        return torch.sigmoid(logits.reshape(-1))
    if task in {"binary", "multiclass"}:
        return torch.softmax(logits, dim=-1)
    raise ValueError(f"Unsupported task_type {task_type!r}.")


def predict_probabilities(
    model: torch.nn.Module,
    inputs: np.ndarray | torch.Tensor,
    *,
    task_type: str,
    device: str | torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    tensor = torch.as_tensor(inputs).float()
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            probabilities.append(probabilities_from_logits(logits, task_type).cpu())
    return torch.cat(probabilities).numpy()


def load_xy_artifact(spec: Mapping[str, Any], base_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if "inputs" not in spec or "labels" not in spec:
        raise ValueError("Data spec requires `inputs` and `labels` artifact specs.")
    return load_array(spec["inputs"], base_dir), load_array(spec["labels"], base_dir)


def build_cooker(spec: Mapping[str, Any], base_dir: Path) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build an authorized cooker from a callable or saved additive perturbation."""
    if "callable" in spec:
        factory = import_object(str(spec["callable"]))
        kwargs = dict(spec.get("kwargs", {}))
        candidate = factory(**kwargs)
        return candidate if callable(candidate) else factory
    if "perturbation" in spec:
        perturbation = torch.as_tensor(load_array(spec["perturbation"], base_dir)).float()
        clamp = spec.get("clamp", [0.0, 1.0])

        def add_perturbation(batch: torch.Tensor) -> torch.Tensor:
            noise = perturbation.to(batch.device)
            if noise.ndim == batch.ndim - 1:
                noise = noise.unsqueeze(0)
            if noise.shape[0] not in (1, batch.shape[0]):
                raise ValueError("Per-sample perturbation count must match the timed batch; use batch_size accordingly.")
            return torch.clamp(batch + noise, float(clamp[0]), float(clamp[1]))

        return add_perturbation
    raise ValueError("Cooker spec requires `callable` or `perturbation`.")
