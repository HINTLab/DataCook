#!/usr/bin/env python3
"""Paired significance tests for the extended historical tabular protocol."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from revision_experiments.common import load_config, resolve_path, setup_logging


DEFAULT_PRIMARY_METRICS = {
    "binary": "acc",
    "multiclass": "acc",
    "multilabel": "subset_acc",
}


def _paired_wilcoxon(a: pd.Series, b: pd.Series) -> dict[str, float | int]:
    paired = pd.concat(
        [pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")],
        axis=1,
    ).dropna()
    difference = paired.iloc[:, 1].to_numpy() - paired.iloc[:, 0].to_numpy()
    nonzero = difference[~np.isclose(difference, 0.0)]
    if len(paired) < 2:
        return {
            "num_pairs": len(paired), "num_nonzero": len(nonzero),
            "statistic": float("nan"), "p_value": float("nan"),
            "mean_difference_b_minus_a": float("nan"),
            "cohens_dz": float("nan"), "rank_biserial": float("nan"),
        }
    if len(nonzero) == 0:
        statistic, p_value, rank_biserial = 0.0, 1.0, 0.0
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
            result = wilcoxon(difference, alternative="two-sided")
        statistic, p_value = float(result.statistic), float(result.pvalue)
        ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy()
        positive = float(ranks[nonzero > 0].sum())
        negative = float(ranks[nonzero < 0].sum())
        rank_biserial = (positive - negative) / (positive + negative)
    std = float(np.std(difference, ddof=1))
    return {
        "num_pairs": len(paired), "num_nonzero": len(nonzero),
        "statistic": statistic, "p_value": p_value,
        "mean_difference_b_minus_a": float(np.mean(difference)),
        "cohens_dz": float(np.mean(difference) / std) if std > 0 else float("nan"),
        "rank_biserial": rank_biserial,
    }


def _holm_by_family(frame: pd.DataFrame, family_columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    output["p_value_holm"] = np.nan
    for _, indices in output.groupby(family_columns, dropna=False).groups.items():
        valid = output.loc[indices, "p_value"].dropna().sort_values()
        running = 0.0
        total = len(valid)
        for rank, (index, value) in enumerate(valid.items()):
            running = max(running, min(1.0, float(value) * (total - rank)))
            output.loc[index, "p_value_holm"] = running
    output["significant_raw"] = output["p_value"] < 0.05
    output["significant_holm"] = output["p_value_holm"] < 0.05
    return output


def build_tests(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    primary = {
        **DEFAULT_PRIMARY_METRICS,
        **config.get("primary_metrics_by_task", {}),
    }
    selected = frame[
        frame.apply(
            lambda row: row["metric"] == primary.get(str(row["task_type"])),
            axis=1,
        )
    ].copy()
    rows = []
    comparisons = (
        ("clean_vs_unauthorized", "clean", "unauthorized_raw"),
        ("clean_vs_authorized", "clean", "authorized_cooked"),
    )
    for keys, group in selected.groupby(
        ["dataset", "task_type", "epsilon", "metric"], dropna=False
    ):
        for comparison, condition_a, condition_b in comparisons:
            rows.append({
                "dataset": keys[0], "task_type": keys[1],
                "epsilon": keys[2], "metric": keys[3],
                "comparison": comparison, "condition_a": condition_a,
                "condition_b": condition_b, "test_type": "wilcoxon_two_sided",
                **_paired_wilcoxon(group[condition_a], group[condition_b]),
            })
    if not rows:
        raise ValueError("No primary-metric rows were available for significance tests.")
    family_columns = list(config.get("holm_family", ["dataset", "comparison"]))
    return _holm_by_family(pd.DataFrame(rows), family_columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("tabular-legacy-statistics", args.verbose)
    config, base_dir = load_config(args.config)
    input_path = resolve_path(config["significance_input"], base_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Significance input does not exist: {input_path}")
    output_path = Path(config["significance_output"]).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    build_tests(pd.read_csv(input_path), config).to_csv(temporary, index=False)
    temporary.replace(output_path)
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
