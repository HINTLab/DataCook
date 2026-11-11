#!/usr/bin/env python3
"""Task 6: paired t-tests and Wilcoxon signed-rank comparisons."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from analysis.aggregate_results import load_long_results
from revision_experiments.common import load_config, setup_logging, write_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("statistical-tests", args.verbose)
    config, base_dir = load_config(args.config)
    frame = load_long_results(config, base_dir)
    comparisons = config.get("comparisons", [])
    if not comparisons:
        raise ValueError("Statistics config requires at least one method comparison.")
    rows = []
    default_pair = [
        "experiment_type", "dataset", "surrogate_model", "victim_model",
        "seed", "epsilon", "setting", "attack_type", "attack_strength", "metric",
    ]
    default_group = [
        "experiment_type", "dataset", "modality", "task_type",
        "surrogate_model", "victim_model", "epsilon", "setting",
        "attack_type", "attack_strength", "metric",
    ]
    for comparison in comparisons:
        method_a, method_b = comparison["method_a"], comparison["method_b"]
        subset = frame[frame["method"].isin([method_a, method_b])].copy()
        if "experiment_type" in comparison:
            subset = subset[subset["experiment_type"] == comparison["experiment_type"]]
        if "metrics" in comparison:
            subset = subset[subset["metric"].isin(comparison["metrics"])]
        pair_by = [column for column in comparison.get("pair_by", default_pair) if column in subset]
        group_by = [column for column in comparison.get("group_by", default_group) if column in subset]
        for keys, group in subset.groupby(group_by, dropna=False):
            pivot = group.pivot_table(index=pair_by, columns="method", values="value", aggfunc="mean").dropna()
            if method_a not in pivot or method_b not in pivot or len(pivot) < 2:
                logger.warning("Skipping %s vs %s for %s: fewer than two pairs.", method_a, method_b, keys)
                continue
            a, b = pivot[method_a].to_numpy(), pivot[method_b].to_numpy()
            requested = comparison.get("test", "auto")
            test_type = "wilcoxon" if requested == "auto" and len(a) < 30 else ("paired_t" if requested == "auto" else requested)
            if test_type == "wilcoxon":
                if np.allclose(a, b):
                    statistic, p_value = 0.0, 1.0
                else:
                    result = wilcoxon(a, b, alternative=comparison.get("alternative", "two-sided"))
                    statistic, p_value = float(result.statistic), float(result.pvalue)
            elif test_type == "paired_t":
                result = ttest_rel(a, b, alternative=comparison.get("alternative", "two-sided"))
                statistic, p_value = float(result.statistic), float(result.pvalue)
            else:
                raise ValueError(f"Unsupported test {test_type!r}.")
            key_values = keys if isinstance(keys, tuple) else (keys,)
            group_map = dict(zip(group_by, key_values))
            alpha = float(comparison.get("alpha", 0.05))
            rows.append({
                "experiment_type": group_map.get("experiment_type", comparison.get("experiment_type", "all")),
                "dataset": group_map.get("dataset", "all"), "dataset_group": comparison.get("dataset_group", group_map.get("dataset", "all")),
                "modality": group_map.get("modality", "all"), "task_type": group_map.get("task_type", "all"),
                "method": f"{method_a}_vs_{method_b}", "surrogate_model": group_map.get("surrogate_model", "all"),
                "victim_model": group_map.get("victim_model", "all"), "seed": "paired",
                "epsilon": group_map.get("epsilon", "all"), "setting": group_map.get("setting", "all"),
                "attack_type": group_map.get("attack_type", ""), "attack_strength": group_map.get("attack_strength", ""),
                "method_a": method_a, "method_b": method_b,
                "metric": group_map.get("metric", "all"), "test_type": test_type, "num_pairs": len(a),
                "statistic": statistic, "p_value": p_value, "alpha": alpha, "significant": bool(p_value < alpha),
            })
    if not rows:
        raise ValueError("No valid paired comparisons were available; no result file was written.")
    output = config.get("significance_output", "results/statistics/significance_tests.csv")
    logger.info("Wrote %s", write_rows(rows, output))


if __name__ == "__main__":
    main()
