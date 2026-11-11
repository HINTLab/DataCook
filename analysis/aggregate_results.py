#!/usr/bin/env python3
"""Task 6: aggregate long-format revision results across random seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from revision_experiments.common import load_config, resolve_path, setup_logging, write_rows


GROUP_COLUMNS = [
    "experiment_type", "dataset", "modality", "task_type", "method",
    "surrogate_model", "victim_model", "epsilon", "setting",
    "attack_type", "attack_strength", "metric",
    "raw_subset_size",
]


def load_long_results(config, base_dir: Path) -> pd.DataFrame:
    frames = []
    inputs = config.get("inputs", [])
    if not inputs:
        raise ValueError("Statistics config requires a non-empty `inputs` list.")
    for item in inputs:
        path = resolve_path(item["path"], base_dir)
        if not path.is_file():
            raise FileNotFoundError(f"Statistics input does not exist: {path}")
        frame = pd.read_csv(path)
        frame["experiment_type"] = item.get("experiment_type", path.stem)
        if "metric" not in frame:
            raise ValueError(f"Long-format input {path} is missing `metric`.")
        value_columns = item.get("value_columns")
        if value_columns is None:
            value_columns = ["value"] if "value" in frame else [name for name in ("cp", "pp", "residual_utility", "recovery_rate", "mean") if name in frame]
        if not value_columns:
            raise ValueError(f"Could not infer numeric value columns in {path}.")
        for value_column in value_columns:
            if value_column not in frame:
                raise KeyError(f"Configured value column {value_column!r} is missing from {path}.")
            selected = frame.copy()
            if value_column != "value":
                selected["metric"] = value_column + ":" + selected["metric"].astype(str)
            selected["value"] = pd.to_numeric(selected[value_column], errors="coerce")
            frames.append(selected)
    combined = pd.concat(frames, ignore_index=True)
    for column in ("dataset", "modality", "task_type", "method", "seed"):
        if column not in combined:
            raise ValueError(f"Statistics inputs are missing required column {column!r}.")
    for column in ("surrogate_model", "victim_model", "epsilon", "setting", "attack_type", "attack_strength", "raw_subset_size"):
        if column not in combined:
            combined[column] = ""
        else:
            combined[column] = combined[column].fillna("")
    return combined


def aggregate(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for keys, group in frame.groupby(GROUP_COLUMNS, dropna=False):
        values = group["value"].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        margin = float(t.ppf(0.975, values.size - 1) * std / np.sqrt(values.size)) if values.size > 1 else 0.0
        rows.append({
            **dict(zip(GROUP_COLUMNS, keys)), "seed": "aggregate",
            "num_seeds": int(group.loc[group["value"].notna(), "seed"].nunique()),
            "num_observations": int(values.size), "mean": mean, "std": std,
            "ci95_low": mean - margin, "ci95_high": mean + margin,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("aggregate-results", args.verbose)
    config, base_dir = load_config(args.config)
    output = config.get("aggregated_output", "results/statistics/aggregated_results.csv")
    path = write_rows(aggregate(load_long_results(config, base_dir)), output)
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
