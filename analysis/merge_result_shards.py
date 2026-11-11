#!/usr/bin/env python3
"""Safely merge non-overlapping revision-result shards into extended CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from revision_experiments.common import load_config, resolve_path, setup_logging


IDENTITY_COLUMNS = [
    "dataset", "modality", "task_type", "method", "surrogate_model",
    "victim_model", "seed", "epsilon", "setting", "attack_type",
    "attack_strength", "metric", "raw_subset_size",
]


def merge_shards(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("At least one input shard is required.")
    frames = []
    columns: list[str] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Result shard does not exist: {path}")
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"Result shard is empty: {path}")
        frame["_source_shard"] = str(path)
        frames.append(frame)
        columns.extend(column for column in frame.columns if column not in columns)
    combined = pd.concat(
        [frame.reindex(columns=columns) for frame in frames], ignore_index=True
    )
    value_columns = [column for column in columns if column != "_source_shard"]
    combined = combined.drop_duplicates(subset=value_columns).reset_index(drop=True)
    keys = [column for column in IDENTITY_COLUMNS if column in combined]
    normalized = combined[keys].copy()
    for column in keys:
        normalized[column] = normalized[column].fillna("").astype(str)
    conflicts = normalized.duplicated(keys, keep=False)
    if conflicts.any():
        examples = combined.loc[conflicts, keys + ["_source_shard"]].head(10)
        raise ValueError(
            "Conflicting result rows share the same experiment identity:\n"
            + examples.to_string(index=False)
        )
    return combined.drop(columns="_source_shard")


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("merge-result-shards", args.verbose)
    config, base_dir = load_config(args.config)
    groups: dict[str, Any] = config.get("groups", {})
    if not groups:
        raise ValueError("Merge config requires a non-empty `groups` mapping.")
    for name, spec in groups.items():
        inputs = [resolve_path(value, base_dir) for value in spec.get("inputs", [])]
        output = Path(spec["output"]).expanduser()
        if not output.is_absolute():
            output = (Path.cwd() / output).resolve()
        merged = merge_shards(inputs)
        logger.info("Wrote %s rows for %s to %s", len(merged), name, _write(merged, output))


if __name__ == "__main__":
    main()
