"""Shared configuration, artifact and CSV utilities."""

from __future__ import annotations

import json
import logging
import copy
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


BASE_METADATA = [
    "dataset",
    "modality",
    "task_type",
    "method",
    "surrogate_model",
    "victim_model",
    "seed",
    "epsilon",
    "setting",
    "attack_type",
    "attack_strength",
]


def setup_logging(name: str, verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("The top-level YAML value must be a mapping.")
    return config, config_path.parent


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()


def _artifact_parts(spec: str | Mapping[str, Any], default_key: str | None) -> tuple[str, str | None]:
    if isinstance(spec, str):
        if "::" in spec:
            path, key = spec.rsplit("::", 1)
            return path, key
        return spec, default_key
    if not isinstance(spec, Mapping) or "path" not in spec:
        raise TypeError("Artifact specs must be a path string or a mapping containing `path`.")
    return str(spec["path"]), spec.get("key", default_key)


def load_array(
    spec: str | Mapping[str, Any],
    base_dir: Path,
    *,
    default_key: str | None = None,
) -> np.ndarray:
    path_text, key = _artifact_parts(spec, default_key)
    path = resolve_path(path_text, base_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key is None:
                if len(archive.files) != 1:
                    raise KeyError(f"Artifact {path} has multiple arrays; specify `key` or use path::key.")
                key = archive.files[0]
            if key not in archive:
                raise KeyError(f"Array {key!r} not found in {path}; available: {archive.files}.")
            return np.asarray(archive[key])
    if suffix == ".csv":
        frame = pd.read_csv(path)
        if key is None:
            if frame.shape[1] != 1:
                raise KeyError(f"CSV {path} has multiple columns; specify `key`.")
            key = str(frame.columns[0])
        if key not in frame:
            raise KeyError(f"Column {key!r} not found in {path}.")
        return frame[key].to_numpy()
    if suffix in {".pt", ".pth"}:
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before weights_only was introduced
            value = torch.load(path, map_location="cpu")
        if key is not None:
            if not isinstance(value, Mapping) or key not in value:
                raise KeyError(f"Key {key!r} not found in tensor artifact {path}.")
            value = value[key]
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)
    raise ValueError(f"Unsupported artifact format {suffix!r} for {path}.")


def load_prediction_artifact(
    spec: str | Mapping[str, Any], base_dir: Path
) -> dict[str, np.ndarray | None]:
    """Load y_true plus y_pred/y_prob from a safe NPZ artifact."""
    if isinstance(spec, str):
        path = resolve_path(spec, base_dir)
        keys: Mapping[str, str] = {}
    else:
        path = resolve_path(str(spec["path"]), base_dir)
        keys = spec.get("keys", {})
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Prediction artifacts must be .npz files: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Prediction artifact does not exist: {path}")
    true_key = keys.get("y_true", "y_true")
    pred_key = keys.get("y_pred", "y_pred")
    prob_key = keys.get("y_prob", "y_prob")
    with np.load(path, allow_pickle=False) as archive:
        if true_key not in archive:
            raise KeyError(f"{path} is missing required array {true_key!r}.")
        if pred_key not in archive and prob_key not in archive:
            raise KeyError(f"{path} must contain {pred_key!r} or {prob_key!r}.")
        return {
            "y_true": np.asarray(archive[true_key]),
            "y_pred": np.asarray(archive[pred_key]) if pred_key in archive else None,
            "y_prob": np.asarray(archive[prob_key]) if prob_key in archive else None,
        }


def metadata(record: Mapping[str, Any], *, setting: str = "", attack_type: str = "", attack_strength: Any = "") -> dict[str, Any]:
    source = record.get("metadata", record)
    row = {field: source.get(field, "") for field in BASE_METADATA}
    row["setting"] = setting or row["setting"]
    row["attack_type"] = attack_type or row["attack_type"]
    row["attack_strength"] = attack_strength if attack_strength != "" else row["attack_strength"]
    missing = [field for field in BASE_METADATA[:8] if row[field] == ""]
    if missing:
        raise ValueError(f"Experiment metadata is missing required fields: {missing}.")
    return row


def write_rows(rows: Iterable[Mapping[str, Any]], output: str | Path, base_dir: Path | None = None) -> Path:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to create an empty result file: {output}")
    path = Path(output).expanduser()
    if not path.is_absolute():
        # Result paths are conventionally relative to the repository/current cwd,
        # while artifact paths are relative to the config file.
        path = (base_dir or Path.cwd()) / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    ordered = [column for column in BASE_METADATA if column in frame.columns]
    ordered += [column for column in frame.columns if column not in ordered]
    frame[ordered].to_csv(path, index=False)
    return path


def json_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def require_records(config: Mapping[str, Any], key: str = "experiments") -> list[dict[str, Any]]:
    records = config.get(key)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Configuration must contain a non-empty `{key}` list.")
    expanded = []
    defaults = config.get("metadata_defaults", {})
    for original in records:
        record = copy.deepcopy(original)
        record_meta = {**defaults, **record.get("metadata", {})}
        seeds = record.pop("seeds", config.get("seeds", [record_meta.get("seed", "")]))
        epsilons = record.pop("epsilons", [record_meta.get("epsilon", "")])
        if not isinstance(seeds, list):
            seeds = [seeds]
        if not isinstance(epsilons, list):
            epsilons = [epsilons]
        for seed in seeds:
            for epsilon in epsilons:
                item = copy.deepcopy(record)
                item["metadata"] = {**record_meta, "seed": seed, "epsilon": epsilon}
                context = {
                    "seed": seed,
                    "epsilon": epsilon,
                    "epsilon_pixel": float(epsilon) * 255 if epsilon != "" else "",
                }
                expanded.append(_format_templates(item, context))
    return expanded


class _KeepUnknown(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _format_templates(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(_KeepUnknown(context))
    if isinstance(value, list):
        return [_format_templates(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _format_templates(item, context) for key, item in value.items()}
    return value


def run_prepare_commands(config: Mapping[str, Any], base_dir: Path, logger: logging.Logger) -> None:
    """Run explicitly configured artifact-generation commands without a shell."""
    commands = list(config.get("prepare_commands", []))
    # Per-record commands are expanded with the same seed/epsilon templates as
    # artifact paths, enabling actual multi-seed and budget generation runs.
    if isinstance(config.get("experiments"), list) and config.get("experiments"):
        for record in require_records(config):
            commands.extend(record.get("prepare_commands", []))
    for index, item in enumerate(commands):
        if isinstance(item, list):
            command, cwd = item, Path.cwd()
        elif isinstance(item, Mapping):
            command = item.get("command")
            cwd = resolve_path(item.get("cwd", "."), base_dir)
        else:
            raise TypeError("prepare_commands entries must be argument lists or mappings.")
        if not isinstance(command, list) or not command:
            raise ValueError(f"prepare_commands[{index}].command must be a non-empty argument list.")
        logger.info("Preparing artifacts: %s", " ".join(map(str, command)))
        subprocess.run([str(part) for part in command], cwd=cwd, check=True)
