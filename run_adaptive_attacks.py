#!/usr/bin/env python3
"""Task 3: evaluate purification, augmentation and fine-tuning attacks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from attacks import fine_tune_model, gaussian_smooth, jpeg_compress, median_filter, strong_augment
from metrics import compute_classification_metrics
from revision_experiments.common import load_config, load_prediction_artifact, metadata, require_records, setup_logging, write_rows
from revision_experiments.modeling import load_model, load_xy_artifact, predict_probabilities


def _apply_attack(images: np.ndarray, attack_type: str, strength: Any, seed: int) -> np.ndarray:
    attack_type = {"gaussian": "gaussian_smoothing", "smoothing": "gaussian_smoothing", "augmentation": "strong_augmentation"}.get(attack_type, attack_type)
    if attack_type == "jpeg":
        return np.asarray(jpeg_compress(images, quality=int(strength)))
    if attack_type == "gaussian_smoothing":
        params = strength if isinstance(strength, dict) else {"sigma": float(strength), "kernel_size": 5}
        return np.asarray(gaussian_smooth(images, **params))
    if attack_type == "strong_augmentation":
        return np.asarray(strong_augment(images, strength=int(strength), seed=seed))
    if attack_type == "median_filter":
        return np.asarray(median_filter(images, kernel_size=int(strength)))
    raise ValueError(f"Unsupported inference attack type {attack_type!r}.")


def _select_raw_subset(
    inputs: np.ndarray,
    labels: np.ndarray,
    subset_size: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic class-stratified raw subset for an attack budget."""
    if len(inputs) != len(labels):
        raise ValueError("Raw subset inputs and labels must have the same length.")
    if subset_size is None:
        return inputs, labels
    subset_size = int(subset_size)
    if subset_size <= 0 or subset_size > len(inputs):
        raise ValueError(
            f"subset_size must be in [1, {len(inputs)}], got {subset_size}."
        )
    targets = np.asarray(labels)
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets.reshape(-1)
    if targets.ndim != 1:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(inputs), size=subset_size, replace=False))
    else:
        classes = np.unique(targets)
        if subset_size < len(classes):
            raise ValueError(
                f"subset_size={subset_size} cannot cover all {len(classes)} classes."
            )
        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=subset_size, random_state=seed
        )
        try:
            indices, _ = next(splitter.split(np.zeros(len(targets)), targets))
        except ValueError as exc:
            raise ValueError(
                f"Could not draw a stratified raw subset of size {subset_size}: {exc}"
            ) from exc
        indices = np.sort(indices)
    return np.asarray(inputs)[indices], np.asarray(labels)[indices]


def _attacked_metrics(record: dict[str, Any], attack: dict[str, Any], base_dir, device: str) -> dict[str, float]:
    task_type = record["metadata"]["task_type"]
    threshold = float(record.get("threshold", 0.5))
    if "predictions" in attack:
        artifact = load_prediction_artifact(attack["predictions"], base_dir)
        return compute_classification_metrics(artifact["y_true"], artifact["y_pred"], artifact["y_prob"], task_type, threshold)

    model = load_model(attack.get("model", record.get("model", {})), base_dir, device)
    attack_type = attack["type"]
    if attack_type in {"finetune", "robust_finetune"}:
        subset_x, subset_y = load_xy_artifact(attack["raw_subset"], base_dir)
        seed = int(record["metadata"]["seed"])
        subset_x, subset_y = _select_raw_subset(
            subset_x, subset_y, attack.get("subset_size"), seed
        )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(torch.as_tensor(subset_x).float(), torch.as_tensor(subset_y)),
            batch_size=int(attack.get("batch_size", 64)), shuffle=True, generator=generator,
        )
        input_transform = None
        if attack_type == "robust_finetune":
            strength = int(attack.get("augmentation_strength", 2))
            input_transform = lambda batch: strong_augment(batch, strength=strength)
        model = fine_tune_model(
            model, loader, task_type=task_type, epochs=int(attack.get("epochs", 5)),
            learning_rate=float(attack.get("learning_rate", 1e-4)), device=device,
            input_transform=input_transform,
            seed=seed,
        )
        eval_x, eval_y = load_xy_artifact(attack["eval_data"], base_dir)
    else:
        eval_x, eval_y = load_xy_artifact(attack.get("eval_data", record.get("eval_data", {})), base_dir)
        eval_x = _apply_attack(eval_x, attack_type, attack["strength"], int(record["metadata"]["seed"]))
    prob = predict_probabilities(model, eval_x, task_type=task_type, device=device, batch_size=int(attack.get("batch_size", 128)))
    return compute_classification_metrics(eval_y, y_prob=prob, task_type=task_type, threshold=threshold)


def _result_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _merge_results(new_output: Path, merge_with: str | Path, combined_output: str | Path) -> Path:
    """Merge old inference attacks and new fine-tuning attacks without changing either input."""
    existing_path = _result_path(merge_with)
    if not existing_path.is_file():
        raise FileNotFoundError(f"Existing adaptive-attack result does not exist: {existing_path}")
    frames = [pd.read_csv(existing_path), pd.read_csv(new_output)]
    columns = list(frames[0].columns) + [
        column for column in frames[1].columns if column not in frames[0].columns
    ]
    combined = pd.concat(
        [frame.reindex(columns=columns) for frame in frames], ignore_index=True
    )
    combined = combined.drop_duplicates().reset_index(drop=True)
    destination = _result_path(combined_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("adaptive-attacks", args.verbose)
    config, base_dir = load_config(args.config)
    rows = []
    for record in require_records(config):
        baseline = {}
        for setting in ("clean", "authorized_cooked", "unauthorized_raw"):
            artifact = load_prediction_artifact(record["predictions"][setting], base_dir)
            baseline[setting] = compute_classification_metrics(
                artifact["y_true"], artifact["y_pred"], artifact["y_prob"],
                record["metadata"]["task_type"], float(record.get("threshold", 0.5)),
            )
        for attack in record.get("attacks", []):
            if "type" not in attack:
                raise ValueError("Every adaptive attack requires `type`.")
            if "strength" in attack:
                strength = attack["strength"]
            else:
                strength = f"epochs={attack.get('epochs', 5)},lr={attack.get('learning_rate', 1e-4):g}"
                if attack["type"] == "robust_finetune":
                    strength += f",aug={attack.get('augmentation_strength', 2)}"
                if "subset_size" in attack:
                    strength += f",n={int(attack['subset_size'])}"
            logger.info("Running %s strength=%s", attack["type"], strength)
            attacked = _attacked_metrics(record, attack, base_dir, args.device)
            metric_names = attack.get("metric_names", record.get("metric_names")) or sorted(set(baseline["clean"]) & set(attacked))
            for metric in metric_names:
                clean = baseline["clean"][metric]
                authorized = baseline["authorized_cooked"][metric]
                unauthorized = baseline["unauthorized_raw"][metric]
                after = attacked[metric]
                denominator = clean - unauthorized
                raw_recovery = (after - unauthorized) / denominator if denominator != 0 else float("nan")
                recovery = float(np.clip(raw_recovery, 0.0, 1.0)) if np.isfinite(raw_recovery) else float("nan")
                rows.append({
                    **metadata(record, setting="adaptive_attack", attack_type=attack["type"], attack_strength=strength),
                    "metric": metric, "clean": clean, "authorized_cooked": authorized,
                    "unauthorized_raw": unauthorized, "attacked_unauthorized_raw": after,
                    "cp_before_attack": unauthorized - clean, "cp_after_attack": after - clean,
                    "recovery_rate": recovery, "recovery_rate_unclipped": raw_recovery,
                    "raw_subset_size": attack.get("subset_size", "full")
                    if attack["type"] in {"finetune", "robust_finetune"} else "",
                })
    if not rows:
        raise ValueError("No attacks were configured; no results were written.")
    output = config.get("output", "results/attacks/adaptive_attack_results.csv")
    output_path = write_rows(rows, output)
    logger.info("Wrote %s", output_path)
    merge_with = config.get("merge_with")
    combined_output = config.get("combined_output")
    if bool(merge_with) != bool(combined_output):
        raise ValueError("`merge_with` and `combined_output` must be configured together.")
    if merge_with and combined_output:
        logger.info("Wrote merged adaptive attacks to %s", _merge_results(output_path, merge_with, combined_output))


if __name__ == "__main__":
    main()
