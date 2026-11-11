#!/usr/bin/env python3
"""Build final DataCook revision statistics and paper-ready CSV tables."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analysis.aggregate_results import GROUP_COLUMNS, aggregate
from revision_experiments.common import load_config, resolve_path, setup_logging


IDENTITY = [
    "dataset", "modality", "task_type", "method", "surrogate_model",
    "victim_model", "seed", "epsilon",
]


def _read_input(config: dict[str, Any], base_dir: Path, name: str, *, optional: bool = False) -> pd.DataFrame:
    value = config.get("inputs", {}).get(name)
    if not value:
        if optional:
            return pd.DataFrame()
        raise ValueError(f"Final-analysis config is missing inputs.{name}.")
    path = resolve_path(value, base_dir)
    if not path.is_file():
        if optional:
            return pd.DataFrame()
        raise FileNotFoundError(f"Required final-analysis input does not exist: {path}")
    return pd.read_csv(path)


def _write(frame: pd.DataFrame, path: Path) -> Path:
    if frame.empty:
        raise ValueError(f"Refusing to write an empty final-analysis table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def build_epsilon_ablation(cp_pp: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    quality_columns = [
        column for column in quality.columns
        if column.endswith("_mean") or column.endswith("_std") or column == "fid"
    ]
    keys = [column for column in IDENTITY if column in cp_pp and column in quality]
    quality_unique = quality[keys + quality_columns].drop_duplicates(keys)
    result = cp_pp.merge(quality_unique, on=keys, how="left", validate="many_to_one")
    result["epsilon_pixel"] = (pd.to_numeric(result["epsilon"]) * 255).round(6)
    return result


def build_budget_all(
    attacks: pd.DataFrame,
    budget: pd.DataFrame,
    full_subset_sizes: dict[str, int],
) -> pd.DataFrame:
    full = attacks[
        attacks["attack_type"].isin(["finetune", "robust_finetune"])
        & attacks["attack_strength"].astype(str).str.contains("epochs=10", regex=False)
        & ~attacks["attack_strength"].astype(str).str.contains(",n=", regex=False)
    ].copy()
    full["raw_subset_size"] = full["dataset"].map(full_subset_sizes)
    full["raw_subset_budget"] = "full"
    frames = [full]
    if not budget.empty:
        selected = budget.copy()
        selected["raw_subset_size"] = pd.to_numeric(selected["raw_subset_size"], errors="raise").astype(int)
        selected["raw_subset_budget"] = selected["raw_subset_size"].astype(str)
        frames.append(selected)
    result = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()
    sort_columns = [column for column in ["dataset", "seed", "raw_subset_size", "attack_type", "metric"] if column in result]
    return result.sort_values(sort_columns).reset_index(drop=True)


def _long_values(frame: pd.DataFrame, experiment_type: str, value_columns: list[str]) -> list[pd.DataFrame]:
    rows = []
    for value_column in value_columns:
        if value_column not in frame:
            continue
        selected = frame.copy()
        if "metric" in selected:
            selected["metric"] = value_column + ":" + selected["metric"].astype(str)
        else:
            selected["metric"] = value_column
        selected["value"] = pd.to_numeric(selected[value_column], errors="coerce")
        selected["experiment_type"] = experiment_type
        rows.append(selected)
    return rows


def build_aggregates(
    cp_pp: pd.DataFrame,
    attacks: pd.DataFrame,
    budget_all: pd.DataFrame,
    quality: pd.DataFrame,
    runtime: pd.DataFrame,
    pseudo: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    frames += _long_values(cp_pp, "main", ["cp", "pp", "residual_utility"])
    frames += _long_values(attacks, "adaptive_attack", [
        "attacked_unauthorized_raw", "cp_after_attack", "recovery_rate",
        "recovery_rate_unclipped",
    ])
    frames += _long_values(budget_all, "finetune_budget", [
        "attacked_unauthorized_raw", "cp_after_attack", "recovery_rate",
        "recovery_rate_unclipped",
    ])
    frames += _long_values(quality, "image_quality", [
        "ssim_mean", "psnr_mean", "lpips_mean", "l2_norm_mean", "linf_norm_mean",
    ])
    frames += _long_values(runtime, "runtime", [
        "cooking_time_per_sample", "raw_inference_time_per_sample",
        "cooked_inference_time_per_sample", "overhead_ratio", "gpu_memory_mb",
    ])
    frames += _long_values(pseudo, "pseudo_labels", [
        "pseudo_acc", "mean_confidence", "cp_acc", "pp_acc", "cp_auc", "pp_auc",
    ])
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column in GROUP_COLUMNS:
        if column not in combined:
            combined[column] = ""
        elif column not in {"epsilon", "raw_subset_size"}:
            combined[column] = combined[column].fillna("")
    combined["raw_subset_size"] = combined["raw_subset_size"].fillna("")
    return pd.DataFrame(aggregate(combined))


def _paired_test(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    paired = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(paired) < 2:
        return float("nan"), float("nan"), float("nan")
    difference = paired.iloc[:, 1].to_numpy() - paired.iloc[:, 0].to_numpy()
    if np.allclose(difference, 0):
        return 0.0, 1.0, 0.0
    with warnings.catch_warnings():
        # With three seeds, zero differences make SciPy use its documented
        # normal approximation. We expose num_pairs and effect size explicitly.
        warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
        result = wilcoxon(paired.iloc[:, 0], paired.iloc[:, 1], alternative="two-sided")
    return float(result.statistic), float(result.pvalue), float(np.mean(difference))


def _holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["p_value_holm"] = np.nan
    for _, indices in frame.groupby("comparison", dropna=False).groups.items():
        valid = frame.loc[indices, "p_value"].dropna().sort_values()
        running = 0.0
        m = len(valid)
        for rank, (index, value) in enumerate(valid.items()):
            running = max(running, min(1.0, float(value) * (m - rank)))
            frame.loc[index, "p_value_holm"] = running
    frame["significant"] = frame["p_value"] < 0.05
    frame["significant_holm"] = frame["p_value_holm"] < 0.05
    return frame


def _primary_subset(frame: pd.DataFrame, settings: dict[str, Any], *, attacks: bool = False) -> pd.DataFrame:
    selected = frame.copy()
    metrics = settings.get("metrics")
    if metrics:
        selected = selected[selected["metric"].isin(metrics)]
    epsilon_map = settings.get("epsilons_pixel_by_dataset", {})
    if epsilon_map:
        mask = pd.Series(False, index=selected.index)
        for dataset, epsilon_pixels in epsilon_map.items():
            epsilon_values = np.asarray(epsilon_pixels, dtype=float) / 255.0
            dataset_values = pd.to_numeric(selected["epsilon"], errors="coerce").to_numpy()
            matches = np.any(np.isclose(dataset_values[:, None], epsilon_values[None, :]), axis=1)
            mask |= selected["dataset"].eq(dataset) & matches
        selected = selected[mask]
    if attacks:
        attack_types = settings.get("attack_types")
        if attack_types:
            selected = selected[selected["attack_type"].isin(attack_types)]
        epochs = settings.get("epochs")
        if epochs:
            pattern = "|".join(f"epochs={int(epoch)}(?:,|$)" for epoch in epochs)
            selected = selected[selected["attack_strength"].astype(str).str.contains(pattern, regex=True)]
    return selected


def build_significance(
    cp_pp: pd.DataFrame,
    attacks: pd.DataFrame,
    budget_all: pd.DataFrame,
    settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    settings = settings or {}
    cp_pp = _primary_subset(cp_pp, settings)
    attacks = _primary_subset(attacks, settings, attacks=True)
    budget_all = _primary_subset(budget_all, settings, attacks=True)
    rows: list[dict[str, Any]] = []
    for keys, group in cp_pp.groupby(["dataset", "epsilon", "metric"], dropna=False):
        statistic, p_value, difference = _paired_test(group["clean"], group["unauthorized_raw"])
        rows.append({
            "comparison": "clean_vs_unauthorized", "dataset": keys[0], "epsilon": keys[1],
            "attack_type": "", "attack_strength": "", "raw_subset_size": "",
            "metric": keys[2], "condition_a": "clean", "condition_b": "unauthorized_raw",
            "num_pairs": int(group[["clean", "unauthorized_raw"]].dropna().shape[0]),
            "test_type": "wilcoxon", "statistic": statistic, "p_value": p_value,
            "mean_difference_b_minus_a": difference,
        })
    attack_sources = [("attack_before_vs_after", attacks), ("budget_before_vs_after", budget_all)]
    for comparison, source in attack_sources:
        group_columns = ["dataset", "epsilon", "attack_type", "attack_strength", "metric"]
        if comparison.startswith("budget"):
            group_columns.insert(-1, "raw_subset_size")
        for keys, group in source.groupby(group_columns, dropna=False):
            values = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
            statistic, p_value, difference = _paired_test(
                group["unauthorized_raw"], group["attacked_unauthorized_raw"]
            )
            rows.append({
                "comparison": comparison, "dataset": values["dataset"],
                "epsilon": values["epsilon"], "attack_type": values["attack_type"],
                "attack_strength": values["attack_strength"],
                "raw_subset_size": values.get("raw_subset_size", ""), "metric": values["metric"],
                "condition_a": "unauthorized_raw", "condition_b": "after_attack",
                "num_pairs": int(group[["unauthorized_raw", "attacked_unauthorized_raw"]].dropna().shape[0]),
                "test_type": "wilcoxon", "statistic": statistic, "p_value": p_value,
                "mean_difference_b_minus_a": difference,
            })
    paired_attack = budget_all.pivot_table(
        index=["dataset", "epsilon", "seed", "raw_subset_size", "metric"],
        columns="attack_type", values="attacked_unauthorized_raw", aggfunc="mean",
    ).reset_index()
    if {"finetune", "robust_finetune"}.issubset(paired_attack.columns):
        for keys, group in paired_attack.groupby(["dataset", "epsilon", "raw_subset_size", "metric"], dropna=False):
            statistic, p_value, difference = _paired_test(group["finetune"], group["robust_finetune"])
            rows.append({
                "comparison": "finetune_vs_robust", "dataset": keys[0], "epsilon": keys[1],
                "attack_type": "finetune_vs_robust_finetune", "attack_strength": "epochs=10",
                "raw_subset_size": keys[2], "metric": keys[3], "condition_a": "finetune",
                "condition_b": "robust_finetune", "num_pairs": len(group), "test_type": "wilcoxon",
                "statistic": statistic, "p_value": p_value,
                "mean_difference_b_minus_a": difference,
            })
    return _holm_adjust(pd.DataFrame(rows))


def _summary_table(frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]) -> pd.DataFrame:
    def safe_mean(values: pd.Series) -> float:
        array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        if not len(array):
            return float("nan")
        if np.isposinf(array).any() and not np.isneginf(array).any():
            return float("inf")
        if np.isneginf(array).any() and not np.isposinf(array).any():
            return float("-inf")
        if np.isinf(array).any():
            return float("nan")
        return float(np.mean(array))

    def safe_std(values: pd.Series) -> float:
        array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        if len(array) <= 1:
            return 0.0
        if np.isinf(array).any():
            return 0.0 if np.all(array == array[0]) else float("nan")
        return float(np.std(array, ddof=1))

    safe_mean.__name__ = "mean"
    safe_std.__name__ = "std"
    selected = [column for column in value_columns if column in frame]
    grouped = frame.groupby(group_columns, dropna=False)[selected].agg([safe_mean, safe_std, "count"])
    grouped.columns = [f"{value}_{stat}" for value, stat in grouped.columns]
    return grouped.reset_index()


def build_tables(
    cp_pp: pd.DataFrame,
    attacks: pd.DataFrame,
    budget_all: pd.DataFrame,
    quality: pd.DataFrame,
    runtime: pd.DataFrame,
    pseudo: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    main = cp_pp[cp_pp["metric"].eq("acc")].copy()
    main["epsilon_pixel"] = (main["epsilon"] * 255).round(6)
    attack_acc = attacks[attacks["metric"].eq("acc")]
    attack_acc = attack_acc.copy()
    attack_acc["epsilon_pixel"] = (attack_acc["epsilon"] * 255).round(6)
    budget_acc = budget_all[budget_all["metric"].eq("acc")].copy()
    budget_acc["epsilon_pixel"] = (budget_acc["epsilon"] * 255).round(6)
    quality = quality.copy()
    quality["epsilon_pixel"] = (quality["epsilon"] * 255).round(6)
    return {
        "table_main_acc.csv": _summary_table(
            main, ["dataset", "epsilon", "epsilon_pixel"],
            ["clean", "authorized_cooked", "unauthorized_raw", "cp", "pp"],
        ),
        "table_adaptive_attacks_acc.csv": _summary_table(
            attack_acc, ["dataset", "epsilon", "epsilon_pixel", "attack_type", "attack_strength"],
            ["unauthorized_raw", "attacked_unauthorized_raw", "cp_before_attack",
             "cp_after_attack", "recovery_rate", "recovery_rate_unclipped"],
        ),
        "table_finetune_budget_acc.csv": _summary_table(
            budget_acc, ["dataset", "epsilon", "epsilon_pixel", "attack_type", "raw_subset_size", "raw_subset_budget"],
            ["unauthorized_raw", "attacked_unauthorized_raw", "cp_after_attack",
             "recovery_rate", "recovery_rate_unclipped"],
        ),
        "table_image_quality.csv": _summary_table(
            quality, ["dataset", "epsilon", "epsilon_pixel"],
            ["ssim_mean", "psnr_mean", "lpips_mean", "l2_norm_mean", "linf_norm_mean"],
        ),
        "table_runtime.csv": _summary_table(
            runtime, ["dataset", "resolution", "device", "batch_size"],
            ["cooking_time_per_sample", "raw_inference_time_per_sample",
             "cooked_inference_time_per_sample", "overhead_ratio", "gpu_memory_mb"],
        ),
        "table_pseudo_labels.csv": _summary_table(
            pseudo, ["dataset"], ["pseudo_acc", "mean_confidence", "cp_acc", "pp_acc", "cp_auc", "pp_auc"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("final-analysis", args.verbose)
    config, base_dir = load_config(args.config)
    cp_pp = _read_input(config, base_dir, "cp_pp")
    quality = _read_input(config, base_dir, "quality")
    attacks = _read_input(config, base_dir, "attacks")
    budget = _read_input(config, base_dir, "budget_attacks", optional=True)
    runtime = _read_input(config, base_dir, "runtime")
    pseudo = _read_input(config, base_dir, "pseudo")
    budget_all = build_budget_all(attacks, budget, config.get("full_subset_sizes", {}))
    output_dir = Path(config.get("output_dir", "results/revision_pipeline/final")).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    outputs = {
        "epsilon_ablation.csv": build_epsilon_ablation(cp_pp, quality),
        "adaptive_finetune_budget_all.csv": budget_all,
        "aggregated_results.csv": build_aggregates(cp_pp, attacks, budget_all, quality, runtime, pseudo),
        "significance_tests.csv": build_significance(
            cp_pp, attacks, budget_all, config.get("significance", {})
        ),
        **build_tables(cp_pp, attacks, budget_all, quality, runtime, pseudo),
    }
    for filename, frame in outputs.items():
        logger.info("Wrote %s", _write(frame, output_dir / filename))


if __name__ == "__main__":
    main()
