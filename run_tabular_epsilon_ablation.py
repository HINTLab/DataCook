#!/usr/bin/env python3
"""Self-contained DataCook epsilon ablation for tabular classification.

Binary, multiclass and multilabel datasets share the same experiment protocol.
Epsilon is measured in standardized-feature L-infinity units and must not be
interpreted as an image pixel budget.
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from metrics import compute_classification_metrics, compute_cp_pp
from revision_experiments.common import load_config, write_rows


PROJECT_ROOT = Path(__file__).resolve().parent
SETTINGS = ("clean", "authorized_cooked", "unauthorized_raw")


class TabularNet(nn.Module):
    """Small MLP matching the architecture used by the legacy tabular scripts."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(previous, hidden), nn.ReLU()))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve(path: str | Path, base_dir: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return ((base_dir or PROJECT_ROOT) / candidate).resolve()


def _normalise_task_type(value: str) -> str:
    task_type = value.lower().replace("-", "").replace("_", "")
    aliases = {
        "binary": "binary",
        "binaryclass": "binary",
        "multiclass": "multiclass",
        "multilabel": "multilabel",
    }
    if task_type not in aliases:
        raise ValueError(
            f"Unsupported tabular task_type {value!r}; expected binary, "
            "multiclass, or multilabel."
        )
    return aliases[task_type]


def load_tabular_split(
    data_path: Path,
    *,
    features: list[str],
    targets: list[str],
    task_type: str,
    seed: int,
    test_size: float,
    categorical_mappings: dict[str, dict[Any, Any]] | None = None,
    target_mapping: dict[Any, Any] | None = None,
    stratify_split: bool = True,
    preprocess_on_full_data: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load, split and standardize a tabular dataset."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Tabular dataset does not exist: {data_path}. Pass --data-path "
            "or update dataset.data_path in the YAML configuration."
        )
    frame = pd.read_csv(data_path)
    missing = [
        column for column in [*features, *targets]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"{data_path} is missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )

    for column, mapping in (categorical_mappings or {}).items():
        if column not in frame:
            raise ValueError(f"Categorical mapping references unknown column {column!r}.")
        frame[column] = frame[column].map(mapping)

    inputs = frame[features].apply(pd.to_numeric, errors="coerce")
    if task_type == "multilabel":
        labels = (
            frame[targets]
            .apply(pd.to_numeric, errors="raise")
            .astype(int)
            .to_numpy()
        )
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("Multilabel targets must contain only 0 and 1.")
        stratify = None
    else:
        if len(targets) != 1:
            raise ValueError(
                f"{task_type} requires exactly one target column, got {targets}."
            )
        target_values = frame[targets[0]]
        if target_mapping:
            target_values = target_values.map(target_mapping)
        if target_values.isna().any():
            unknown = sorted(frame.loc[target_values.isna(), targets[0]].astype(str).unique())
            raise ValueError(
                f"Target mapping for {targets[0]!r} did not cover values: {unknown}."
            )
        labels = pd.to_numeric(target_values, errors="raise").astype(int).to_numpy()
        unique_labels = sorted(np.unique(labels).tolist())
        expected = list(range(len(unique_labels)))
        if unique_labels != expected:
            raise ValueError(
                f"{targets[0]} labels must be contiguous integers starting at 0; "
                f"found {unique_labels}."
            )
        if task_type == "binary" and unique_labels != [0, 1]:
            raise ValueError(f"{targets[0]} must contain binary labels 0 and 1.")
        stratify = labels if stratify_split else None

    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0,1).")

    if preprocess_on_full_data:
        # Historical compatibility: the original tabular scripts imputed and
        # standardized the complete dataset before the train/test split.
        full_medians = inputs.median(numeric_only=True)
        inputs = inputs.fillna(full_medians)
        if inputs.isna().any().any():
            raise ValueError(
                f"{data_path} contains feature columns that cannot be converted to numbers."
            )
        scaler = StandardScaler()
        # Match the historical scripts: StandardScaler receives pandas' native
        # float64 array, and conversion to torch/float32 happens only afterwards.
        scaled = scaler.fit_transform(inputs.to_numpy()).astype(np.float32)
        x_train, x_test, y_train, y_test = train_test_split(
            scaled,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            inputs,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
        train_medians = x_train.median(numeric_only=True)
        x_train = x_train.fillna(train_medians)
        x_test = x_test.fillna(train_medians)
        if x_train.isna().any().any() or x_test.isna().any().any():
            raise ValueError(
                f"{data_path} contains feature columns that cannot be converted to numbers."
            )
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train.to_numpy(dtype=np.float32)).astype(np.float32)
        x_test = scaler.transform(x_test.to_numpy(dtype=np.float32)).astype(np.float32)
    all_scaled = np.concatenate((x_train, x_test), axis=0)
    return (
        torch.from_numpy(x_train),
        torch.from_numpy(
            y_train.astype(np.float32)
            if task_type == "multilabel"
            else y_train.astype(np.int64)
        ),
        torch.from_numpy(x_test),
        torch.from_numpy(
            y_test.astype(np.float32)
            if task_type == "multilabel"
            else y_test.astype(np.int64)
        ),
        torch.from_numpy(all_scaled.min(axis=0).astype(np.float32)),
        torch.from_numpy(all_scaled.max(axis=0).astype(np.float32)),
    )


def _train_model(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    task_type: str,
    class_weights: torch.Tensor | None = None,
    shuffle: bool = True,
    validation_inputs: torch.Tensor | None = None,
    validation_labels: torch.Tensor | None = None,
    select_best_on_validation: bool = False,
    use_global_data_order: bool = False,
) -> nn.Module:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive.")
    generator = None if use_global_data_order else torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(inputs, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if task_type == "multilabel":
        criterion: nn.Module = nn.BCEWithLogitsLoss(
            pos_weight=class_weights.to(device) if class_weights is not None else None
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )
    if select_best_on_validation and (
        validation_inputs is None or validation_labels is None
    ):
        raise ValueError(
            "select_best_on_validation requires validation inputs and labels."
        )
    best_score = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(epochs):
        model.train()
        for batch_inputs, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs.to(device))
            labels_device = batch_labels.to(device)
            if task_type != "multilabel":
                labels_device = labels_device.reshape(-1)
            loss = criterion(logits, labels_device)
            loss.backward()
            optimizer.step()
        if select_best_on_validation:
            model.eval()
            with torch.no_grad():
                logits = model(validation_inputs.to(device))
                if task_type == "multilabel":
                    predictions = (logits >= 0).to(torch.int64).cpu()
                    truth = validation_labels.to(torch.int64).cpu()
                    score = float((predictions == truth).all(dim=1).float().mean())
                else:
                    predictions = logits.argmax(dim=1).cpu()
                    truth = validation_labels.reshape(-1).to(torch.int64).cpu()
                    score = float((predictions == truth).float().mean())
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def _probabilities(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    task_type: str = "binary",
) -> np.ndarray:
    loader = DataLoader(TensorDataset(inputs), batch_size=batch_size, shuffle=False)
    batches = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            probability = (
                torch.sigmoid(logits)
                if task_type == "multilabel"
                else torch.softmax(logits, dim=1)
            )
            batches.append(probability.cpu())
    return torch.cat(batches).numpy()


def cook_tabular_inputs(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    epsilon: float,
    attackable_indices: list[int],
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    num_steps: int,
    device: torch.device,
    step_size_ratio: float | None = None,
    step_size: float | None = None,
    task_type: str = "binary",
) -> tuple[torch.Tensor, float]:
    """Generate DataCook error-minimizing perturbations for a whole split."""
    if epsilon <= 0:
        raise ValueError("Tabular epsilon must be positive.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if step_size is not None and step_size_ratio is not None:
        raise ValueError("Configure either step_size or step_size_ratio, not both.")
    if step_size is None and step_size_ratio is None:
        raise ValueError("Either step_size or step_size_ratio must be configured.")
    if step_size is not None and step_size <= 0:
        raise ValueError("step_size must be positive.")
    if step_size_ratio is not None and step_size_ratio <= 0:
        raise ValueError("step_size_ratio must be positive.")
    if not attackable_indices:
        raise ValueError("At least one attackable feature is required.")
    if min(attackable_indices) < 0 or max(attackable_indices) >= inputs.shape[1]:
        raise ValueError("An attackable feature index is outside the input feature range.")

    original = inputs.to(device)
    lower = x_min.to(device)
    upper = x_max.to(device)
    mask = torch.zeros_like(original)
    mask[:, attackable_indices] = 1.0
    model.eval()
    with torch.no_grad():
        logits = model(original)
        pseudo_labels = (
            (logits >= 0).to(logits.dtype)
            if task_type == "multilabel"
            else logits.argmax(dim=1)
        )

    cooked = original.clone().detach()
    effective_step_size = (
        float(step_size)
        if step_size is not None
        else float(epsilon) * float(step_size_ratio)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(num_steps):
        cooked.requires_grad_(True)
        logits = model(cooked)
        loss = (
            nn.functional.binary_cross_entropy_with_logits(logits, pseudo_labels)
            if task_type == "multilabel"
            else nn.functional.cross_entropy(logits, pseudo_labels)
        )
        gradient = torch.autograd.grad(loss, cooked, only_inputs=True)[0]
        with torch.no_grad():
            # DataCook uses the error-minimizing direction (-sign(gradient)).
            cooked = cooked - effective_step_size * gradient.sign() * mask
            delta = torch.clamp(cooked - original, -float(epsilon), float(epsilon))
            cooked = torch.maximum(torch.minimum(original + delta, upper), lower)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return cooked.detach().cpu(), elapsed


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }


def _metadata(
    dataset: str,
    seed: int,
    epsilon: float,
    setting: str,
    task_type: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "modality": "tabular",
        "task_type": task_type,
        "method": "datacook",
        "surrogate_model": "mlp",
        "victim_model": "mlp",
        "seed": seed,
        "epsilon": epsilon,
        "epsilon_scale": "standardized_feature_linf",
        "setting": setting,
        "attack_type": "",
        "attack_strength": "",
    }


def _run_name(dataset: str, seed: int, epsilon: float) -> str:
    return f"{dataset}-datacook-sur=mlp-vic=mlp-seed={seed}-eps={epsilon:g}"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).to_dict(orient="records")


def run_ablation(config: dict[str, Any], base_dir: Path, args) -> dict[str, Path]:
    dataset_cfg = config.get("dataset", {})
    experiment_cfg = config.get("experiment", {})
    model_cfg = config.get("model", {})
    output_cfg = config.get("outputs", {})
    dataset_name = str(dataset_cfg.get("name", "Diabetes"))
    task_type = _normalise_task_type(dataset_cfg.get("task_type", "binary"))
    features = list(dataset_cfg.get("features", []))
    configured_targets = dataset_cfg.get(
        "targets", [dataset_cfg.get("target", "Outcome")]
    )
    targets = (
        [str(configured_targets)]
        if isinstance(configured_targets, str)
        else [str(value) for value in configured_targets]
    )
    if not features:
        raise ValueError("dataset.features must be a non-empty list.")
    if not targets:
        raise ValueError("dataset.target or dataset.targets must be configured.")
    data_path = _resolve(
        args.data_path or dataset_cfg.get("data_path", "TableData/dataset/diabetes.csv"),
        base_dir if not args.data_path else PROJECT_ROOT,
    )
    seeds = list(args.seeds or experiment_cfg.get("seeds", [0, 1, 2]))
    epsilons = list(args.epsilons or experiment_cfg.get("epsilons", [0.1, 0.25, 0.5, 1.0, 2.0]))
    if not seeds or not epsilons:
        raise ValueError("At least one seed and epsilon are required.")
    if len(set(seeds)) != len(seeds) or len(set(epsilons)) != len(epsilons):
        raise ValueError("Seeds and epsilons must not contain duplicates.")

    attackable_features = list(experiment_cfg.get(
        "attackable_features", ["BMI", "DiabetesPedigreeFunction"]
    ))
    missing_attackable = [name for name in attackable_features if name not in features]
    if missing_attackable:
        raise ValueError(f"Unknown attackable features: {missing_attackable}")
    attackable_indices = [features.index(name) for name in attackable_features]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    artifact_root = _resolve(
        getattr(args, "artifact_root", None)
        or output_cfg.get("artifact_root", "artifacts/revision_pipeline_tabular")
    )
    result_root = _resolve(
        getattr(args, "result_root", None)
        or output_cfg.get("result_root", "results/revision_pipeline_tabular")
    )
    requested = len(seeds) * len(epsilons)
    logging.info(
        "Tabular preflight: dataset=%s path=%s rows_expected_runs=%d device=%s",
        dataset_name, data_path, requested, device,
    )
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Missing tabular dataset: {data_path}. "
            "Set TABULAR_DATA in the job or pass --data-path."
        )
    if args.dry_run:
        frame = pd.read_csv(data_path, nrows=5)
        missing = [
            column for column in [*features, *targets]
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(f"Dataset preflight failed; missing columns: {missing}")
        logging.info("Dry run complete; no training or output was performed.")
        return {}

    hidden_dims = tuple(int(value) for value in model_cfg.get("hidden_dims", [64, 32]))
    dropout = float(model_cfg.get("dropout", 0.3))
    configured_baseline_checkpoint = (
        getattr(args, "baseline_checkpoint", None)
        or model_cfg.get("baseline_checkpoint")
    )
    baseline_checkpoint = (
        _resolve(
            configured_baseline_checkpoint,
            PROJECT_ROOT if getattr(args, "baseline_checkpoint", None) else base_dir,
        )
        if configured_baseline_checkpoint else None
    )
    require_baseline_checkpoint = bool(
        model_cfg.get("require_baseline_checkpoint", False)
    )
    if baseline_checkpoint is not None and not baseline_checkpoint.is_file():
        raise FileNotFoundError(
            f"Configured baseline checkpoint does not exist: {baseline_checkpoint}"
        )
    if require_baseline_checkpoint and baseline_checkpoint is None:
        raise ValueError(
            "model.require_baseline_checkpoint=true but no baseline checkpoint "
            "was configured or passed with --baseline-checkpoint."
        )
    baseline_epochs = int(experiment_cfg.get("baseline_epochs", 100))
    protected_epochs = int(experiment_cfg.get("protected_epochs", 100))
    batch_size = int(experiment_cfg.get("batch_size", 32))
    learning_rate = float(experiment_cfg.get("learning_rate", 1.0e-3))
    num_steps = int(experiment_cfg.get("num_steps", 20))
    configured_step_size = experiment_cfg.get("step_size")
    configured_step_size_ratio = experiment_cfg.get("step_size_ratio")
    if configured_step_size is not None and configured_step_size_ratio is not None:
        raise ValueError(
            "experiment.step_size and experiment.step_size_ratio are mutually exclusive."
        )
    if configured_step_size is None and configured_step_size_ratio is None:
        configured_step_size_ratio = 0.25
    step_size = (
        float(configured_step_size)
        if configured_step_size is not None else None
    )
    step_size_ratio = (
        float(configured_step_size_ratio)
        if configured_step_size_ratio is not None else None
    )
    test_size = float(dataset_cfg.get("test_size", 0.2))
    categorical_mappings = dict(dataset_cfg.get("categorical_mappings", {}))
    target_mapping = dataset_cfg.get("target_mapping")
    class_balance = bool(experiment_cfg.get("class_balance", False))
    max_positive_weight = float(experiment_cfg.get("max_positive_weight", 100.0))
    split_seed = experiment_cfg.get("split_seed")
    stratify_split = bool(experiment_cfg.get("stratify_split", True))
    preprocess_on_full_data = bool(
        experiment_cfg.get("preprocess_on_full_data", False)
    )
    baseline_select_best_on_test = bool(
        experiment_cfg.get("baseline_select_best_on_test", False)
    )
    baseline_shuffle = bool(experiment_cfg.get("baseline_shuffle", True))
    protected_shuffle = bool(experiment_cfg.get("protected_shuffle", True))
    configured_victim_seed = experiment_cfg.get("victim_seed")
    configured_protected_seed = experiment_cfg.get("protected_seed")
    victim_seed_offset = int(experiment_cfg.get("victim_seed_offset", 100_000))
    protected_seed_offset = int(experiment_cfg.get("protected_seed_offset", 0))
    reset_rng_each_epsilon = bool(
        experiment_cfg.get("reset_rng_each_epsilon", False)
    )
    victim_initialization_count = int(
        experiment_cfg.get("victim_initialization_count", 1)
    )
    victim_deepcopy_template = bool(
        experiment_cfg.get("victim_deepcopy_template", False)
    )
    historical_loader_rng_draws = int(
        experiment_cfg.get("historical_loader_rng_draws_before_training", 0)
    )
    if historical_loader_rng_draws < 0:
        raise ValueError(
            "experiment.historical_loader_rng_draws_before_training must be non-negative."
        )
    if victim_initialization_count <= 0:
        raise ValueError("experiment.victim_initialization_count must be positive.")
    baseline_global_data_order = bool(
        experiment_cfg.get("baseline_global_data_order", False)
    )
    protected_global_data_order = bool(
        experiment_cfg.get("protected_global_data_order", False)
    )

    metric_rows: list[dict[str, Any]] = []
    cp_rows: list[dict[str, Any]] = []
    epsilon_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed = int(seed)
        split = load_tabular_split(
            data_path,
            features=features,
            targets=targets,
            task_type=task_type,
            seed=int(split_seed) if split_seed is not None else seed,
            test_size=test_size,
            categorical_mappings=categorical_mappings,
            target_mapping=target_mapping,
            stratify_split=stratify_split,
            preprocess_on_full_data=preprocess_on_full_data,
        )
        x_train, y_train, x_test, y_test, x_min, x_max = split
        if task_type == "multilabel":
            num_outputs = len(targets)
        else:
            observed_classes = int(torch.max(torch.cat((y_train, y_test))).item()) + 1
            num_outputs = int(dataset_cfg.get("num_classes", observed_classes))
            if num_outputs < observed_classes:
                raise ValueError(
                    f"dataset.num_classes={num_outputs} is smaller than the "
                    f"{observed_classes} observed classes."
                )

        class_weights = None
        if class_balance:
            if task_type == "multilabel":
                positives = y_train.sum(dim=0)
                negatives = y_train.shape[0] - positives
                if torch.any(positives == 0):
                    raise ValueError(
                        f"{dataset_name} has a training label with no positive samples."
                    )
                class_weights = torch.clamp(
                    negatives / positives,
                    max=max_positive_weight,
                ).to(torch.float32)
            else:
                counts = torch.bincount(y_train, minlength=num_outputs).to(torch.float32)
                if torch.any(counts == 0):
                    raise ValueError(
                        f"{dataset_name} has a training class with no samples."
                    )
                class_weights = y_train.numel() / (num_outputs * counts)

        baseline_dir = artifact_root / f"{dataset_name}-seed={seed}"
        baseline_path = baseline_dir / "clean_model.pth"
        _seed_everything(seed)
        baseline = TabularNet(
            len(features),
            output_dim=num_outputs,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        if baseline_checkpoint is not None:
            checkpoint = torch.load(baseline_checkpoint, map_location="cpu")
            state_dict = (
                checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                else checkpoint
            )
            baseline.load_state_dict(state_dict)
            baseline = baseline.to(device).eval()
            logging.info("Loaded historical baseline: %s", baseline_checkpoint)
            baseline_dir.mkdir(parents=True, exist_ok=True)
            torch.save(_state_dict_cpu(baseline), baseline_path)
        elif baseline_path.is_file() and not args.force:
            baseline.load_state_dict(torch.load(baseline_path, map_location="cpu"))
            baseline = baseline.to(device).eval()
            logging.info("Reusing baseline: %s", baseline_path)
        else:
            baseline = _train_model(
                baseline, x_train, y_train, device=device,
                epochs=baseline_epochs, batch_size=batch_size,
                learning_rate=learning_rate, seed=seed,
                task_type=task_type, class_weights=class_weights,
                shuffle=baseline_shuffle,
                validation_inputs=x_test,
                validation_labels=y_test,
                select_best_on_validation=baseline_select_best_on_test,
                use_global_data_order=baseline_global_data_order,
            )
            baseline_dir.mkdir(parents=True, exist_ok=True)
            torch.save(_state_dict_cpu(baseline), baseline_path)
        clean_prob = _probabilities(
            baseline,
            x_test,
            device=device,
            batch_size=batch_size,
            task_type=task_type,
        )
        clean_metrics = compute_classification_metrics(
            y_test.numpy(), y_prob=clean_prob, task_type=task_type
        )

        victim_seed = (
            int(configured_victim_seed)
            if configured_victim_seed is not None else seed + victim_seed_offset
        )
        protected_seed = (
            int(configured_protected_seed)
            if configured_protected_seed is not None else seed + protected_seed_offset
        )
        _seed_everything(victim_seed)
        victim_template = None
        for _ in range(victim_initialization_count):
            victim_template = TabularNet(
                len(features),
                output_dim=num_outputs,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        assert victim_template is not None
        victim_initial_state = copy.deepcopy(victim_template.state_dict())
        paired_cpu_rng_state = torch.random.get_rng_state()
        paired_cuda_rng_states = (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        )

        for epsilon in epsilons:
            epsilon = float(epsilon)
            if reset_rng_each_epsilon:
                # Epsilon is the only changing factor within a seed. Restoring
                # the post-initialization RNG state prevents budget order from
                # changing dropout masks or shuffled minibatches.
                torch.random.set_rng_state(paired_cpu_rng_state)
                if paired_cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(paired_cuda_rng_states)
            effective_step_size = (
                step_size
                if step_size is not None
                else epsilon * float(step_size_ratio)
            )
            attack_parameters = {
                "num_steps": num_steps,
                "step_size": effective_step_size,
                "step_size_ratio": (
                    step_size_ratio if step_size is None else float("nan")
                ),
            }
            run_name = _run_name(dataset_name, seed, epsilon)
            run_artifacts = artifact_root / run_name
            run_results = result_root / run_name / "metrics"
            per_run_files = {
                "main": run_results / "main_metrics.csv",
                "cp": run_results / "cp_pp_summary.csv",
                "epsilon": run_results / "epsilon_ablation.csv",
                "runtime": run_results / "runtime_summary.csv",
            }
            prediction_files = {
                setting: run_artifacts / "predictions" / f"{setting}.npz"
                for setting in SETTINGS
            }
            ready = all(path.is_file() for path in [
                *per_run_files.values(), *prediction_files.values(),
            ])
            if ready and not args.force:
                logging.info("Reusing completed tabular run: %s", run_name)
                metric_rows.extend(_read_rows(per_run_files["main"]))
                cp_rows.extend(_read_rows(per_run_files["cp"]))
                epsilon_rows.extend(_read_rows(per_run_files["epsilon"]))
                runtime_rows.extend(_read_rows(per_run_files["runtime"]))
                continue

            logging.info("Running %s", run_name)
            cooked_train, train_cooking_time = cook_tabular_inputs(
                baseline, x_train, epsilon=epsilon,
                attackable_indices=attackable_indices,
                x_min=x_min, x_max=x_max, num_steps=num_steps,
                step_size_ratio=step_size_ratio, step_size=step_size,
                device=device,
                task_type=task_type,
            )
            cooked_test, test_cooking_time = cook_tabular_inputs(
                baseline, x_test, epsilon=epsilon,
                attackable_indices=attackable_indices,
                x_min=x_min, x_max=x_max, num_steps=num_steps,
                step_size_ratio=step_size_ratio, step_size=step_size,
                device=device,
                task_type=task_type,
            )
            _atomic_npz(
                run_artifacts / "cooked_train.npz",
                raw=x_train.numpy(), cooked=cooked_train.numpy(),
                labels=y_train.numpy(),
            )
            _atomic_npz(
                run_artifacts / "cooked_test.npz",
                raw=x_test.numpy(), cooked=cooked_test.numpy(),
                labels=y_test.numpy(),
            )

            if victim_deepcopy_template:
                # Historical PrivacyProtector uses copy.deepcopy(raw_model).
                # Unlike constructing and then loading, this does not advance
                # the global RNG before shuffled batches and dropout training.
                victim = copy.deepcopy(victim_template)
            else:
                victim = TabularNet(
                    len(features),
                    output_dim=num_outputs,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                )
                victim.load_state_dict(victim_initial_state)
            # Constructing a DataLoader iterator advances the global torch RNG
            # even with shuffle=False. The old scripts iterated the raw train
            # and test loaders once to obtain pseudo-labels before training.
            for _ in range(historical_loader_rng_draws):
                torch.empty((), dtype=torch.int64).random_()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_started = time.perf_counter()
            victim = _train_model(
                victim, cooked_train, y_train, device=device,
                epochs=protected_epochs, batch_size=batch_size,
                learning_rate=learning_rate, seed=protected_seed,
                task_type=task_type, class_weights=class_weights,
                shuffle=protected_shuffle,
                use_global_data_order=protected_global_data_order,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            protected_training_time = time.perf_counter() - training_started
            torch.save(_state_dict_cpu(victim), run_artifacts / "protected_model.pth")

            inference_times: dict[str, float] = {}
            probabilities: dict[str, np.ndarray] = {"clean": clean_prob}
            for setting, inputs in (
                ("authorized_cooked", cooked_test),
                ("unauthorized_raw", x_test),
            ):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                probabilities[setting] = _probabilities(
                    victim,
                    inputs,
                    device=device,
                    batch_size=batch_size,
                    task_type=task_type,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_times[setting] = (
                    time.perf_counter() - started
                ) / len(inputs)
            for setting, probability in probabilities.items():
                predictions = (
                    (probability >= 0.5).astype(np.int64)
                    if task_type == "multilabel"
                    else np.argmax(probability, axis=1)
                )
                _atomic_npz(
                    prediction_files[setting],
                    y_true=y_test.numpy(),
                    y_pred=predictions,
                    y_prob=probability,
                )

            all_metrics = {"clean": clean_metrics}
            for setting in ("authorized_cooked", "unauthorized_raw"):
                all_metrics[setting] = compute_classification_metrics(
                    y_test.numpy(), y_prob=probabilities[setting],
                    task_type=task_type,
                )
            requested_metrics = sorted(
                set.intersection(*(set(values) for values in all_metrics.values()))
            )
            cp_pp = compute_cp_pp(
                all_metrics["clean"],
                all_metrics["authorized_cooked"],
                all_metrics["unauthorized_raw"],
                requested_metrics,
                task_type=task_type,
                num_classes=num_outputs,
                label_prevalence=(
                    y_test.to(torch.float32).mean(dim=0).tolist()
                    if task_type == "multilabel"
                    else None
                ),
            )
            run_metric_rows = []
            for setting, values in all_metrics.items():
                run_metric_rows.extend({
                    **_metadata(
                        dataset_name, seed, epsilon, setting, task_type
                    ),
                    **attack_parameters,
                    "metric": metric,
                    "value": value,
                } for metric, value in values.items())
            run_cp_rows = [{
                **_metadata(
                    dataset_name, seed, epsilon, "cp_pp", task_type
                ),
                **attack_parameters,
                "metric": metric,
                **values,
            } for metric, values in cp_pp.items()]
            run_epsilon_rows = [{
                **_metadata(
                    dataset_name,
                    seed,
                    epsilon,
                    "epsilon_ablation",
                    task_type,
                ),
                **attack_parameters,
                "metric": metric,
                **values,
            } for metric, values in cp_pp.items()]
            run_runtime_rows = [{
                **_metadata(
                    dataset_name, seed, epsilon, "runtime", task_type
                ),
                "resolution": f"{len(features)}_features",
                "num_samples": len(x_test),
                "num_train_samples": len(x_train),
                "num_test_samples": len(x_test),
                "batch_size": batch_size,
                "device": str(device),
                "attackable_features": ",".join(attackable_features),
                **attack_parameters,
                "train_cooking_time_total": train_cooking_time,
                "train_cooking_time_per_sample": train_cooking_time / len(x_train),
                "test_cooking_time_total": test_cooking_time,
                "test_cooking_time_per_sample": test_cooking_time / len(x_test),
                "cooking_time_total": test_cooking_time,
                "cooking_time_per_sample": test_cooking_time / len(x_test),
                "authorized_cooking_latency_per_sample": test_cooking_time / len(x_test),
                "raw_inference_time_per_sample": inference_times["unauthorized_raw"],
                "cooked_inference_time_per_sample": inference_times["authorized_cooked"],
                "overhead_ratio": (
                    test_cooking_time / len(x_test)
                    + inference_times["authorized_cooked"]
                ) / inference_times["unauthorized_raw"]
                if inference_times["unauthorized_raw"] > 0 else float("nan"),
                "protected_training_time": protected_training_time,
                "gpu_memory_mb": (
                    float(torch.cuda.max_memory_allocated(device) / 1024**2)
                    if device.type == "cuda" else float("nan")
                ),
                "repeats": 1,
                "warmup": 0,
            }]
            write_rows(run_metric_rows, per_run_files["main"])
            write_rows(run_cp_rows, per_run_files["cp"])
            write_rows(run_epsilon_rows, per_run_files["epsilon"])
            write_rows(run_runtime_rows, per_run_files["runtime"])
            metric_rows.extend(run_metric_rows)
            cp_rows.extend(run_cp_rows)
            epsilon_rows.extend(run_epsilon_rows)
            runtime_rows.extend(run_runtime_rows)

    outputs = {
        "main_metrics": result_root / "summary/main_metrics.csv",
        "cp_pp": result_root / "summary/cp_pp_summary.csv",
        "epsilon": result_root / "epsilon_ablation.csv",
        "runtime": result_root / "runtime_summary.csv",
    }
    write_rows(metric_rows, outputs["main_metrics"])
    write_rows(cp_rows, outputs["cp_pp"])
    write_rows(epsilon_rows, outputs["epsilon"])
    write_rows(runtime_rows, outputs["runtime"])
    logging.info("Completed %d tabular runs. Results: %s", requested, result_root)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/revision/tabular_epsilon_diabetes.yaml",
    )
    parser.add_argument("--data-path", help="Override dataset.data_path.")
    parser.add_argument("--artifact-root", help="Override outputs.artifact_root.")
    parser.add_argument("--result-root", help="Override outputs.result_root.")
    parser.add_argument(
        "--baseline-checkpoint",
        help="Load a historical surrogate checkpoint instead of retraining it.",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epsilons", nargs="+", type=float)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | tabular-epsilon | %(message)s",
    )
    config, base_dir = load_config(args.config)
    run_ablation(config, base_dir, args)


if __name__ == "__main__":
    main()
