#!/usr/bin/env python3
"""Task 1/2: unified metrics and generalized CP/PP from prediction artifacts."""

from __future__ import annotations

import argparse

from metrics import compute_classification_metrics, compute_cp_pp
from revision_experiments.common import (
    load_config,
    load_prediction_artifact,
    metadata,
    require_records,
    setup_logging,
    write_rows,
)


SETTINGS = ("clean", "authorized_cooked", "unauthorized_raw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("unified-evaluation", args.verbose)
    config, base_dir = load_config(args.config)
    metric_rows, cp_rows = [], []

    for index, record in enumerate(require_records(config)):
        task_type = record.get("task_type", record.get("metadata", {}).get("task_type"))
        predictions = record.get("predictions", {})
        missing = [setting for setting in SETTINGS if setting not in predictions]
        if missing:
            raise ValueError(f"Experiment {index} is missing prediction settings: {missing}.")
        logger.info("Evaluating %s (%s)", record.get("metadata", {}).get("dataset", index), task_type)
        all_metrics = {}
        clean_y_true = None
        for setting in SETTINGS:
            artifact = load_prediction_artifact(predictions[setting], base_dir)
            if setting == "clean":
                clean_y_true = artifact["y_true"]
            values = compute_classification_metrics(
                artifact["y_true"], artifact["y_pred"], artifact["y_prob"],
                task_type=task_type, threshold=float(record.get("threshold", 0.5)),
            )
            all_metrics[setting] = values
            meta = metadata(record, setting=setting)
            metric_rows.extend({**meta, "metric": name, "value": value} for name, value in values.items())

        requested = record.get("metric_names") or sorted(set.intersection(*(set(values) for values in all_metrics.values())))
        label_prevalence = record.get("label_prevalence")
        normalized_task = str(task_type).replace("-", "").replace("_", "").lower()
        if label_prevalence is None and normalized_task == "multilabel":
            import numpy as np

            truth = np.asarray(clean_y_true)
            if truth.ndim != 2:
                raise ValueError("Multi-label clean y_true must have shape (N,L).")
            label_prevalence = truth.mean(axis=0).tolist()
        cp_pp = compute_cp_pp(
            all_metrics["clean"], all_metrics["authorized_cooked"], all_metrics["unauthorized_raw"], requested,
            record.get("random_guess_metrics"), task_type=task_type,
            num_classes=record.get("num_classes"), label_prevalence=label_prevalence,
        )
        base_meta = metadata(record, setting="cp_pp")
        cp_rows.extend({**base_meta, "metric": name, **values} for name, values in cp_pp.items())

    outputs = config.get("outputs", {})
    metrics_path = write_rows(metric_rows, outputs.get("main_metrics", "results/metrics/main_metrics.csv"))
    cp_path = write_rows(cp_rows, outputs.get("cp_pp", "results/metrics/cp_pp_summary.csv"))
    logger.info("Wrote %s and %s", metrics_path, cp_path)


if __name__ == "__main__":
    main()
