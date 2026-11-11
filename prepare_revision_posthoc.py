#!/usr/bin/env python3
"""Incrementally prepare post-hoc artifacts/configs from completed pipeline runs.

The script never trains a model and never modifies files inside an experiment
run.  It reads existing prediction, pseudo-label, checkpoint and ``ae_test``
artifacts, exports paired NumPy arrays under ``artifacts/``, and writes generated
Task 3/5/7/8 configs.  Existing valid exports are reused unless ``--force`` is
passed.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from run_datacook_suite import load_pipeline_configs


PROJECT_ROOT = Path(__file__).resolve().parent
POSTHOC_TASKS = {
    "pseudo",
    "runtime",
    "image_quality",
    "attacks",
    "finetune_attacks",
    "finetune_budget_attacks",
}
RAW_VALIDATION_TASKS = {"finetune_attacks", "finetune_budget_attacks"}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    modality: str
    task_type: str
    num_classes: int
    surrogate: str
    victim: str
    method: str
    seed: int
    epsilon_pixel: float
    epsilon: float
    run_name: str
    run_dir: Path
    model_config: Path
    data_path: Path
    test_count: int
    high_resolution: bool = False

    @property
    def predictions(self) -> dict[str, Path]:
        root = self.run_dir / "predictions"
        return {
            "clean": root / "clean.npz",
            "authorized_cooked": root / "authorized_cooked.npz",
            "unauthorized_raw": root / "unauthorized_raw.npz",
        }

    @property
    def pseudo_labels(self) -> Path:
        return self.run_dir / self.surrogate / "pseudo_labels_train_dataset.npz"

    @property
    def cooked_dir(self) -> Path:
        return self.run_dir / self.surrogate / "ae_test"

    @property
    def test_perturbation(self) -> Path:
        return self.run_dir / "test_perturbation.pt"

    @property
    def checkpoint(self) -> Path:
        return self.run_dir / "protected" / self.victim / "checkpoints" / f"{self.victim}.pth"

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "modality": self.modality,
            "task_type": self.task_type,
            "method": self.method,
            "surrogate_model": self.surrogate,
            "victim_model": self.victim,
            "seed": self.seed,
            "epsilon": self.epsilon,
        }


def _resolve(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", str(value)).strip("-")


def _run_name(dataset: str, method: str, surrogate: str, victim: str, seed: int, epsilon: float) -> str:
    return _slug(
        f"{dataset}-{method}-sur={surrogate}-vic={victim}-seed={seed}-eps={epsilon:g}"
    )


def load_run_specs(
    suite_path: Path,
    selected_datasets: set[str] | None = None,
    selected_seeds: set[int] | None = None,
    selected_epsilons: set[float] | None = None,
) -> list[RunSpec]:
    """Expand the runs produced by a suite invocation.

    ``run_datacook_suite.py --seeds/--epsilons`` overrides the values stored in
    each pipeline YAML.  Mirror that behavior here so post-hoc preparation can
    discover runs created with command-line overrides (for example seeds 3-7
    when the base YAML contains seeds 0-2).
    """
    specs: list[RunSpec] = []
    for pipeline_path in load_pipeline_configs(suite_path):
        config = yaml.safe_load(pipeline_path.read_text(encoding="utf-8")) or {}
        dataset_cfg = config["dataset"]
        dataset_name = str(dataset_cfg["name"])
        if selected_datasets and dataset_name not in selected_datasets:
            continue
        models = config["models"]
        experiment = config.get("experiment", {})
        surrogate = str(models.get("surrogate", "resnet18"))
        victim = str(models.get("victim", surrogate))
        method = str(experiment.get("attack_type", "datacook"))
        scale = str(experiment.get("epsilon_scale", "pixel"))
        experiment_root = _resolve(config.get("experiment_root", "experiments/revision_pipeline"))
        config_path = _resolve(dataset_cfg["config_path"])
        data_path = _resolve(dataset_cfg.get("data_path", "../datasets"))
        test_shape = dataset_cfg.get("test_noise_shape") or []
        if not test_shape:
            raise ValueError(f"{pipeline_path} has no dataset.test_noise_shape.")
        seed_values = (
            sorted(selected_seeds)
            if selected_seeds is not None
            else config.get("seeds", [0])
        )
        epsilon_values = (
            sorted(selected_epsilons)
            if selected_epsilons is not None
            else config.get("epsilons", [experiment.get("epsilon", 8)])
        )
        for seed in seed_values:
            for epsilon_pixel in epsilon_values:
                epsilon_pixel = float(epsilon_pixel)
                normalized = epsilon_pixel / 255.0 if scale == "pixel" else epsilon_pixel
                run_name = _run_name(
                    dataset_name, method, surrogate, victim, int(seed), epsilon_pixel
                )
                specs.append(RunSpec(
                    dataset=dataset_name,
                    modality=str(dataset_cfg.get("modality", "2d_image")),
                    task_type=str(dataset_cfg.get("task_type", "multiclass")),
                    num_classes=int(dataset_cfg["num_classes"]),
                    surrogate=surrogate,
                    victim=victim,
                    method=method,
                    seed=int(seed),
                    epsilon_pixel=epsilon_pixel,
                    epsilon=float(normalized),
                    run_name=run_name,
                    run_dir=experiment_root / run_name,
                    model_config=config_path / f"{victim}.yaml",
                    data_path=data_path,
                    test_count=int(test_shape[0]),
                    high_resolution=bool(dataset_cfg.get("high_resolution", False)),
                ))
    if not specs:
        raise ValueError("No runs matched the suite configuration and command-line overrides.")
    return specs


def _load_official_split(spec: RunSpec, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an official MedMNIST split with the same preprocessing as dataset.py."""
    import medmnist
    from medmnist import INFO

    gray_2d = {
        "OrganAMNIST", "OrganCMNIST", "OrganSMNIST", "OCTMNIST",
        "PneumoniaMNIST", "BreastMNIST", "TissueMNIST", "ChestMNIST",
    }
    info = INFO[spec.dataset.lower()]
    data_class = getattr(medmnist, info["python_class"])
    # dataset.py intentionally uses MedMNIST's default root for official data;
    # keeping the same constructor here guarantees identical split ordering.
    if spec.modality == "3d_volume":
        from util import Transform3D

        transform = Transform3D()
        as_rgb = False
    else:
        from torchvision import transforms

        transform = transforms.Compose([transforms.ToTensor()])
        as_rgb = spec.dataset in gray_2d
    dataset_kwargs = {
        "split": split,
        "transform": transform,
        "download": True,
        "as_rgb": as_rgb,
    }
    if spec.high_resolution:
        dataset_kwargs["size"] = 224
    dataset = data_class(
        **dataset_kwargs,
    )
    images, labels = [], []
    for image, label in dataset:
        array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
        images.append(np.asarray(array, dtype=np.float32))
        label_array = np.asarray(label)
        if spec.task_type.replace("-", "").replace("_", "").lower() == "multilabel":
            labels.append(label_array.reshape(-1).astype(np.int64))
        else:
            labels.append(int(label_array.reshape(-1)[0]))
    stacked = np.stack(images)
    targets = np.asarray(labels, dtype=np.int64)
    if split == "test" and len(stacked) != spec.test_count:
        raise ValueError(
            f"{spec.dataset} test count mismatch: config={spec.test_count}, loaded={len(stacked)}."
        )
    return stacked, targets


def _load_official_test_set(spec: RunSpec) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for the official test split."""
    return _load_official_split(spec, "test")


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _validate_npz(path: Path, expected_count: int, required_keys: tuple[str, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return all(key in archive and len(archive[key]) == expected_count for key in required_keys)
    except (OSError, ValueError, KeyError):
        return False


def _validate_xy_npz(path: Path, expected_count: int | None = None) -> bool:
    """Return whether an NPZ contains a non-empty, aligned images/labels pair."""
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "images" not in archive or "labels" not in archive:
                return False
            count = len(archive["images"])
            return count > 0 and len(archive["labels"]) == count and (
                expected_count is None or count == expected_count
            )
    except (OSError, ValueError, KeyError):
        return False


def export_raw_test(
    spec: RunSpec,
    output: Path,
    *,
    force: bool,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    if output.is_file() and not force:
        if not _validate_npz(output, spec.test_count, ("images", "labels")):
            raise ValueError(f"Existing raw export is invalid; rerun with --force: {output}")
        with np.load(output, allow_pickle=False) as archive:
            return np.asarray(archive["images"]), np.asarray(archive["labels"])
    logger.info("Loading official %s test split.", spec.dataset)
    images, labels = _load_official_test_set(spec)
    _atomic_npz(output, images=images, labels=labels)
    logger.info("Exported raw test artifact: %s", output)
    return images, labels


def export_raw_validation(
    spec: RunSpec,
    output: Path,
    *,
    force: bool,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """Export the official validation split used by adaptive fine-tuning attacks."""
    if output.is_file() and not force:
        if not _validate_xy_npz(output):
            raise ValueError(f"Existing validation export is invalid; rerun with --force: {output}")
        with np.load(output, allow_pickle=False) as archive:
            return np.asarray(archive["images"]), np.asarray(archive["labels"])
    logger.info("Loading official %s validation split.", spec.dataset)
    images, labels = _load_official_split(spec, "val")
    _atomic_npz(output, images=images, labels=labels)
    logger.info("Exported raw validation artifact: %s", output)
    return images, labels


def _cooked_index(path: Path) -> int:
    suffix = path.stem.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"Cooked filename does not end in a global sample index: {path}")
    return int(suffix)


def export_cooked_test(
    cooked_dir: Path,
    raw_images: np.ndarray,
    raw_labels: np.ndarray,
    output: Path,
    *,
    force: bool,
    logger: logging.Logger,
    perturbation_path: Path | None = None,
) -> None:
    count = len(raw_images)
    if output.is_file() and not force:
        if not _validate_npz(output, count, ("images", "labels")):
            raise ValueError(f"Existing cooked export is invalid; rerun with --force: {output}")
        logger.info("Reusing cooked test artifact: %s", output)
        return
    cooked_npz = cooked_dir / "perturb_data.npz"
    if cooked_npz.is_file():
        with np.load(cooked_npz, allow_pickle=False) as archive:
            if "perturb_images" not in archive or "perturb_labels" not in archive:
                raise ValueError(
                    f"Cooked NPZ must contain perturb_images and perturb_labels: {cooked_npz}"
                )
            cooked_images = np.asarray(archive["perturb_images"], dtype=np.float32)
            cooked_labels = np.asarray(archive["perturb_labels"])
        if len(cooked_images) != count or cooked_images.shape != raw_images.shape:
            raise ValueError(
                f"Cooked/raw shape mismatch for {cooked_npz}: "
                f"{cooked_images.shape} != {raw_images.shape}."
            )
        if cooked_labels.shape != raw_labels.shape or not np.array_equal(cooked_labels, raw_labels):
            raise ValueError(
                f"Cooked/raw labels differ for {cooked_npz}: "
                f"{cooked_labels.shape} != {raw_labels.shape}."
            )
        _atomic_npz(output, images=cooked_images, labels=cooked_labels)
        logger.info("Exported cooked NPZ artifact: %s", output)
        return
    paths = [path for path in cooked_dir.glob("*/*") if path.is_file() and path.suffix.lower() == ".png"]
    if len(paths) != count:
        if perturbation_path is None or not perturbation_path.is_file():
            raise ValueError(
                f"Expected {count} cooked PNGs in {cooked_dir}, found {len(paths)}, "
                "and no test perturbation is available for recovery."
            )
        logger.warning(
            "Expected %d cooked PNGs in %s, found %d; reconstructing from %s.",
            count,
            cooked_dir,
            len(paths),
            perturbation_path,
        )
        import torch

        try:
            perturbation = torch.load(
                perturbation_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            perturbation = torch.load(perturbation_path, map_location="cpu")
        if torch.is_tensor(perturbation):
            perturbation = perturbation.detach().cpu().numpy()
        perturbation = np.asarray(perturbation, dtype=np.float32)
        if perturbation.shape != raw_images.shape:
            raise ValueError(
                f"Cannot reconstruct cooked test images: perturbation shape "
                f"{perturbation.shape} != raw shape {raw_images.shape}."
            )
        cooked_float = np.clip(
            np.asarray(raw_images, dtype=np.float32) + perturbation, 0.0, 1.0
        )
        # torchvision.save_image uses x*255 + 0.5 followed by uint8
        # conversion.  Reproduce that quantization so recovered arrays match
        # the PNG inputs used by the protected-model evaluation.
        cooked_images = (
            np.floor(cooked_float * 255.0 + 0.5)
            .clip(0, 255)
            .astype(np.uint8)
            .astype(np.float32)
            / 255.0
        )
        _atomic_npz(output, images=cooked_images, labels=raw_labels)
        logger.info("Recovered cooked test artifact: %s", output)
        return
    ordered: list[Path | None] = [None] * count
    for path in paths:
        index = _cooked_index(path)
        if index >= count:
            raise ValueError(f"Cooked sample index {index} is outside [0,{count}) in {path}.")
        if ordered[index] is not None:
            raise ValueError(f"Duplicate cooked sample index {index}: {ordered[index]} and {path}.")
        ordered[index] = path
        folder_label = int(path.parent.name)
        if folder_label != int(raw_labels[index]):
            raise ValueError(
                f"Cooked/raw label mismatch at sample {index}: {folder_label} != {raw_labels[index]}."
            )
    missing = [index for index, path in enumerate(ordered) if path is None]
    if missing:
        raise ValueError(f"Cooked samples are missing global indices: {missing[:20]}")
    cooked = []
    for path in ordered:
        assert path is not None
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        cooked.append(np.transpose(image, (2, 0, 1)))
    cooked_images = np.stack(cooked).astype(np.float32, copy=False)
    if cooked_images.shape != raw_images.shape:
        raise ValueError(
            f"Cooked/raw shape mismatch for {cooked_dir}: {cooked_images.shape} != {raw_images.shape}."
        )
    _atomic_npz(output, images=cooked_images, labels=raw_labels)
    logger.info("Exported cooked test artifact: %s", output)


def _relative(path: Path, config_dir: Path) -> str:
    return Path(os.path.relpath(path, config_dir)).as_posix()


def _prediction_record(spec: RunSpec, config_dir: Path) -> dict[str, Any]:
    return {
        "metadata": spec.metadata(),
        "num_classes": spec.num_classes,
        "predictions": {
            key: _relative(path, config_dir) for key, path in spec.predictions.items()
        },
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_configs(
    specs: list[RunSpec],
    artifact_root: Path,
    config_dir: Path,
    *,
    attacks_epsilon: float,
    runtime_seed: int,
    runtime_epsilon: float,
    output_suffix: str = "",
    finetune_epochs: tuple[int, ...] = (5, 10),
    budget_sizes: tuple[int, ...] = (25, 50),
    tasks: set[str] | None = None,
    result_root: str | Path = "results/revision_pipeline",
) -> dict[str, int]:
    selected_tasks = set(POSTHOC_TASKS if tasks is None else tasks)
    unknown_tasks = selected_tasks - POSTHOC_TASKS
    if unknown_tasks:
        raise ValueError(f"Unknown post-hoc tasks: {sorted(unknown_tasks)}")
    suffix = f"_{_slug(output_suffix)}" if output_suffix else ""
    result_root = Path(result_root).as_posix().rstrip("/")
    if not finetune_epochs or any(epoch <= 0 for epoch in finetune_epochs):
        raise ValueError("finetune_epochs must contain positive integers.")
    if not budget_sizes or any(size <= 0 for size in budget_sizes):
        raise ValueError("budget_sizes must contain positive integers.")
    attack_output = f"{result_root}/posthoc/adaptive_attack_results{suffix}.csv"
    finetune_output = f"{result_root}/posthoc/adaptive_finetune_results{suffix}.csv"
    combined_attack_output = f"{result_root}/posthoc/adaptive_attack_results_all{suffix}.csv"
    budget_output = f"{result_root}/posthoc/adaptive_finetune_budget_results{suffix}.csv"
    pseudo_samples_output = f"{result_root}/posthoc/pseudo_label_samples{suffix}.csv"
    pseudo_quality_output = f"{result_root}/posthoc/pseudo_label_quality{suffix}.csv"
    image_quality_output = f"{result_root}/posthoc/quality_metrics{suffix}.csv"
    runtime_output = f"{result_root}/posthoc/runtime_summary{suffix}.csv"
    pseudo_records, quality_records, runtime_records = [], [], []
    attack_records, finetune_records, finetune_budget_records = [], [], []
    for spec in specs:
        raw_path = artifact_root / spec.dataset / "raw_test.npz"
        raw_val_path = artifact_root / spec.dataset / "raw_val.npz"
        cooked_path = artifact_root / spec.run_name / "cooked_test.npz"
        predictions_ready = all(path.is_file() for path in spec.predictions.values())
        base = _prediction_record(spec, config_dir)
        if (
            "pseudo" in selected_tasks
            and predictions_ready
            and spec.pseudo_labels.is_file()
        ):
            pseudo_records.append({
                **base,
                "pseudo_labels": _relative(spec.pseudo_labels, config_dir),
            })
        if _validate_npz(raw_path, spec.test_count, ("images", "labels")) and _validate_npz(
            cooked_path, spec.test_count, ("images", "labels")
        ):
            if "image_quality" in selected_tasks:
                quality_records.append({
                    "metadata": spec.metadata(),
                    "images": {
                        "raw": _relative(raw_path, config_dir) + "::images",
                        "cooked": _relative(cooked_path, config_dir) + "::images",
                    },
                    "compute_lpips": spec.modality != "3d_volume",
                })
            if (
                "runtime" in selected_tasks
                and
                spec.seed == runtime_seed
                and np.isclose(spec.epsilon_pixel, runtime_epsilon)
                and spec.checkpoint.is_file()
                and spec.model_config.is_file()
                and spec.test_perturbation.is_file()
            ):
                runtime_records.append({
                    "metadata": spec.metadata(),
                    "model": {
                        "config_path": _relative(spec.model_config, config_dir),
                        "checkpoint": _relative(spec.checkpoint, config_dir),
                    },
                    "data": {
                        "raw_inputs": _relative(raw_path, config_dir) + "::images",
                        "cooked_inputs": _relative(cooked_path, config_dir) + "::images",
                    },
                    "cooker": {
                        "perturbation": _relative(spec.test_perturbation, config_dir),
                        "clamp": [0, 1],
                    },
                    "batch_size": 16 if (
                        spec.high_resolution or spec.modality == "3d_volume"
                    ) else 128,
                    "repeats": 5,
                    "warmup": 1,
                })
        if (
            selected_tasks & {
                "attacks", "finetune_attacks", "finetune_budget_attacks"
            }
            and
            predictions_ready
            and _validate_npz(raw_path, spec.test_count, ("images", "labels"))
            and np.isclose(spec.epsilon_pixel, attacks_epsilon)
            and spec.checkpoint.is_file()
            and spec.model_config.is_file()
        ):
            if "attacks" in selected_tasks:
                attack_records.append({
                    **base,
                    "model": {
                        "config_path": _relative(spec.model_config, config_dir),
                        "checkpoint": _relative(spec.checkpoint, config_dir),
                    },
                    "eval_data": {
                        "inputs": _relative(raw_path, config_dir) + "::images",
                        "labels": _relative(raw_path, config_dir) + "::labels",
                    },
                    "attacks": [
                        {"type": "jpeg", "strength": quality} for quality in (95, 75, 50)
                    ] + [
                        {"type": "gaussian_smoothing", "strength": {"sigma": sigma, "kernel_size": 5}}
                        for sigma in (0.5, 1.0, 2.0)
                    ] + [
                        {"type": "strong_augmentation", "strength": strength}
                        for strength in (1, 2, 3)
                    ] + [
                        {"type": "median_filter", "strength": kernel} for kernel in (3, 5)
                    ],
                })
            if (
                selected_tasks & {"finetune_attacks", "finetune_budget_attacks"}
                and _validate_xy_npz(raw_val_path)
            ):
                with np.load(raw_val_path, allow_pickle=False) as archive:
                    raw_val_count = len(archive["labels"])
                raw_subset = {
                    "inputs": _relative(raw_val_path, config_dir) + "::images",
                    "labels": _relative(raw_val_path, config_dir) + "::labels",
                }
                eval_data = {
                    "inputs": _relative(raw_path, config_dir) + "::images",
                    "labels": _relative(raw_path, config_dir) + "::labels",
                }
                if "finetune_attacks" in selected_tasks:
                    finetune_records.append({
                        **base,
                        "model": {
                            "config_path": _relative(spec.model_config, config_dir),
                            "checkpoint": _relative(spec.checkpoint, config_dir),
                        },
                        "eval_data": eval_data,
                        "attacks": [
                            {
                                "type": "finetune",
                                "epochs": epochs,
                                "learning_rate": 1.0e-4,
                                "batch_size": 16 if spec.high_resolution else 64,
                                "raw_subset": raw_subset,
                                "eval_data": eval_data,
                            }
                            for epochs in finetune_epochs
                        ] + [
                            {
                                "type": "robust_finetune",
                                "epochs": epochs,
                                "learning_rate": 1.0e-4,
                                "batch_size": 16 if spec.high_resolution else 64,
                                "augmentation_strength": 2,
                                "raw_subset": raw_subset,
                                "eval_data": eval_data,
                            }
                            for epochs in finetune_epochs
                        ],
                    })
                budget_attacks = [
                    {
                        "type": attack_type,
                        "epochs": 10,
                        "learning_rate": 1.0e-4,
                            "batch_size": 16 if spec.high_resolution else 64,
                        "subset_size": subset_size,
                        "raw_subset": raw_subset,
                        "eval_data": eval_data,
                        **({"augmentation_strength": 2} if attack_type == "robust_finetune" else {}),
                    }
                    for subset_size in budget_sizes
                    if subset_size <= raw_val_count
                    for attack_type in ("finetune", "robust_finetune")
                ]
                if (
                    "finetune_budget_attacks" in selected_tasks
                    and budget_attacks
                ):
                    finetune_budget_records.append({
                        **base,
                        "model": {
                            "config_path": _relative(spec.model_config, config_dir),
                            "checkpoint": _relative(spec.checkpoint, config_dir),
                        },
                        "eval_data": eval_data,
                        "attacks": budget_attacks,
                    })

    full_subset_sizes: dict[str, int] = {}
    for spec in specs:
        raw_val_path = artifact_root / spec.dataset / "raw_val.npz"
        if spec.dataset not in full_subset_sizes and _validate_xy_npz(raw_val_path):
            with np.load(raw_val_path, allow_pickle=False) as archive:
                full_subset_sizes[spec.dataset] = len(archive["labels"])

    payloads = {
        "pseudo.yaml": {
            "experiments": pseudo_records,
            "outputs": {
                "samples": pseudo_samples_output,
                "quality": pseudo_quality_output,
            },
        },
        "image_quality.yaml": {
            "experiments": quality_records,
            "output": image_quality_output,
        },
        "runtime.yaml": {
            "experiments": runtime_records,
            "include_average": True,
            "output": runtime_output,
        },
        "attacks.yaml": {
            "experiments": attack_records,
            "output": attack_output,
        },
        "attacks_finetune.yaml": {
            "experiments": finetune_records,
            "output": finetune_output,
            "merge_with": attack_output,
            "combined_output": combined_attack_output,
        },
        "attacks_finetune_budget.yaml": {
            "experiments": finetune_budget_records,
            "output": budget_output,
        },
        "final_analysis.yaml": {
            "inputs": {
                "cp_pp": _relative(PROJECT_ROOT / result_root / "summary/cp_pp_summary.csv", config_dir),
                "quality": _relative(PROJECT_ROOT / image_quality_output, config_dir),
                "attacks": _relative(PROJECT_ROOT / combined_attack_output, config_dir),
                "budget_attacks": _relative(PROJECT_ROOT / budget_output, config_dir),
                "runtime": _relative(PROJECT_ROOT / runtime_output, config_dir),
                "pseudo": _relative(PROJECT_ROOT / pseudo_quality_output, config_dir),
            },
            "full_subset_sizes": full_subset_sizes,
            "output_dir": f"{result_root}/final{suffix}",
        },
    }
    for filename, payload in payloads.items():
        _write_yaml(config_dir / filename, payload)

    master_path = config_dir.parent / "generated.yaml"
    relative_config_dir = Path(os.path.relpath(config_dir, master_path.parent)).as_posix()
    _write_yaml(master_path, {
        "tasks": {
            "metrics": {"enabled": False, "config": "metrics.yaml"},
            "attacks": {"enabled": bool(attack_records), "config": f"{relative_config_dir}/attacks.yaml"},
            "finetune_attacks": {"enabled": False, "config": f"{relative_config_dir}/attacks_finetune.yaml"},
            "finetune_budget_attacks": {"enabled": False, "config": f"{relative_config_dir}/attacks_finetune_budget.yaml"},
            "epsilon": {"enabled": False, "config": "epsilon.yaml"},
            "pseudo": {"enabled": bool(pseudo_records), "config": f"{relative_config_dir}/pseudo.yaml"},
            "statistics": {"enabled": False, "config": "statistics.yaml"},
            "runtime": {"enabled": bool(runtime_records), "config": f"{relative_config_dir}/runtime.yaml"},
            "image_quality": {"enabled": bool(quality_records), "config": f"{relative_config_dir}/image_quality.yaml"},
            "final_analysis": {"enabled": False, "config": f"{relative_config_dir}/final_analysis.yaml"},
        }
    })
    return {
        "pseudo": len(pseudo_records),
        "image_quality": len(quality_records),
        "runtime": len(runtime_records),
        "attacks": len(attack_records),
        "finetune_attacks": len(finetune_records),
        "finetune_budget_attacks": len(finetune_budget_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", default="configs/pipelines/medmnist_three.yaml")
    parser.add_argument("--artifact-root", default="artifacts/revision_pipeline")
    parser.add_argument("--result-root", default="results/revision_pipeline")
    parser.add_argument("--config-dir", default="configs/revision/generated")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(POSTHOC_TASKS),
        help=(
            "Prepare only the selected post-hoc tasks. By default all task "
            "configs are prepared for backward compatibility."
        ),
    )
    parser.add_argument("--datasets", nargs="+", help="Optional dataset-name filter.")
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        help="Override the seed list in every selected pipeline, matching run_datacook_suite.py.",
    )
    parser.add_argument(
        "--epsilons", nargs="+", type=float,
        help="Override the pixel-scale epsilon list in every selected pipeline.",
    )
    parser.add_argument("--attacks-epsilon", type=float, default=8.0)
    parser.add_argument("--runtime-seed", type=int, default=0)
    parser.add_argument("--runtime-epsilon", type=float, default=8.0)
    parser.add_argument("--output-suffix", default="", help="Append a safe suffix to generated post-hoc result filenames.")
    parser.add_argument("--finetune-epochs", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--budget-sizes", nargs="+", type=int, default=[25, 50])
    parser.add_argument("--force", action="store_true", help="Re-export derived NPZ files only; training artifacts remain read-only.")
    parser.add_argument("--dry-run", action="store_true", help="Inventory expected runs without writing or loading datasets.")
    parser.add_argument("--strict", action="store_true", help="Fail if any expected primary artifact is missing.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("revision-posthoc")
    suite_path = _resolve(args.suite_config)
    artifact_root = _resolve(args.artifact_root)
    config_dir = _resolve(args.config_dir)
    selected_tasks = set(args.tasks or POSTHOC_TASKS)
    specs = load_run_specs(
        suite_path,
        set(args.datasets) if args.datasets else None,
        set(args.seeds) if args.seeds else None,
        set(args.epsilons) if args.epsilons else None,
    )
    logger.info("Discovered %d expected runs across %d datasets.", len(specs), len({item.dataset for item in specs}))

    manifest_rows: list[dict[str, Any]] = []
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    raw_validation_exported: set[str] = set()
    missing_primary_count = 0
    for spec in specs:
        runtime_candidate = (
            "runtime" in selected_tasks
            and spec.seed == args.runtime_seed
            and np.isclose(spec.epsilon_pixel, args.runtime_epsilon)
        )
        quality_candidate = "image_quality" in selected_tasks
        attack_candidate = (
            bool(selected_tasks & {
                "attacks", "finetune_attacks", "finetune_budget_attacks",
            })
            and np.isclose(spec.epsilon_pixel, args.attacks_epsilon)
        )
        needs_raw_test = runtime_candidate or quality_candidate or attack_candidate
        needs_cooked_test = runtime_candidate or quality_candidate
        required: dict[str, Path] = {}
        if "pseudo" in selected_tasks or attack_candidate:
            required.update({
                f"prediction_{key}": path
                for key, path in spec.predictions.items()
            })
        if "pseudo" in selected_tasks:
            required["pseudo_labels"] = spec.pseudo_labels
        if needs_cooked_test:
            required["test_perturbation"] = spec.test_perturbation
        if runtime_candidate or attack_candidate:
            required["checkpoint"] = spec.checkpoint
            required["model_config"] = spec.model_config
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            missing_primary_count += 1
            logger.warning("%s missing: %s", spec.run_name, ", ".join(missing))
        raw_path = artifact_root / spec.dataset / "raw_test.npz"
        raw_val_path = artifact_root / spec.dataset / "raw_val.npz"
        cooked_path = artifact_root / spec.run_name / "cooked_test.npz"
        if (
            not args.dry_run
            and attack_candidate
            and selected_tasks & RAW_VALIDATION_TASKS
            and spec.dataset not in raw_validation_exported
        ):
            export_raw_validation(spec, raw_val_path, force=args.force, logger=logger)
            raw_validation_exported.add(spec.dataset)
        if (
            not args.dry_run
            and needs_raw_test
            and spec.dataset not in raw_cache
        ):
            raw_cache[spec.dataset] = export_raw_test(
                spec, raw_path, force=args.force, logger=logger
            )
        if (
            not args.dry_run
            and needs_cooked_test
            and (spec.cooked_dir.is_dir() or spec.test_perturbation.is_file())
        ):
            if spec.dataset not in raw_cache:
                raw_cache[spec.dataset] = export_raw_test(
                    spec, raw_path, force=args.force, logger=logger
                )
            raw_images, raw_labels = raw_cache[spec.dataset]
            export_cooked_test(
                spec.cooked_dir,
                raw_images,
                raw_labels,
                cooked_path,
                force=args.force,
                logger=logger,
                perturbation_path=spec.test_perturbation,
            )
        manifest_rows.append({
            "dataset": spec.dataset,
            "seed": spec.seed,
            "epsilon_pixel": spec.epsilon_pixel,
            "epsilon": spec.epsilon,
            "run_name": spec.run_name,
            "primary_complete": not missing,
            "missing_primary": ";".join(missing),
            "raw_export": str(raw_path),
            "raw_validation_export": str(raw_val_path),
            "cooked_export": str(cooked_path),
        })

    if args.dry_run:
        logger.info("Dry run complete: %d/%d expected runs have missing primary artifacts.", missing_primary_count, len(specs))
        if args.strict and missing_primary_count:
            raise SystemExit(2)
        return

    _write_manifest(artifact_root / "manifest.csv", manifest_rows)
    counts = generate_configs(
        specs,
        artifact_root,
        config_dir,
        attacks_epsilon=args.attacks_epsilon,
        runtime_seed=args.runtime_seed,
        runtime_epsilon=args.runtime_epsilon,
        output_suffix=args.output_suffix,
        finetune_epochs=tuple(args.finetune_epochs),
        budget_sizes=tuple(args.budget_sizes),
        tasks=selected_tasks,
        result_root=args.result_root,
    )
    logger.info("Generated configs: %s", ", ".join(f"{key}={value}" for key, value in counts.items()))
    logger.info("Manifest: %s", artifact_root / "manifest.csv")
    logger.info("Master config: %s", config_dir.parent / "generated.yaml")
    if args.strict and missing_primary_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
