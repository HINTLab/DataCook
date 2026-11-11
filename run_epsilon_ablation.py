#!/usr/bin/env python3
"""Task 4: evaluate configurable perturbation-budget runs."""

from __future__ import annotations

import argparse

from metrics import compute_classification_metrics, compute_cp_pp, compute_image_quality_metrics
from revision_experiments.common import load_array, load_config, load_prediction_artifact, metadata, require_records, setup_logging, write_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("epsilon-ablation", args.verbose)
    config, base_dir = load_config(args.config)
    rows = []
    for record in require_records(config):
        task = record["metadata"]["task_type"]
        values = {}
        for setting in ("clean", "authorized_cooked", "unauthorized_raw"):
            artifact = load_prediction_artifact(record["predictions"][setting], base_dir)
            values[setting] = compute_classification_metrics(
                artifact["y_true"], artifact["y_pred"], artifact["y_prob"], task, float(record.get("threshold", 0.5)),
            )
        metric_names = record.get("metric_names") or sorted(set.intersection(*(set(item) for item in values.values())))
        cp_pp = compute_cp_pp(
            values["clean"], values["authorized_cooked"], values["unauthorized_raw"], metric_names,
            record.get("random_guess_metrics"), task_type=task, num_classes=record.get("num_classes"),
            label_prevalence=record.get("label_prevalence"),
        )
        quality = {name: float("nan") for name in ("ssim_mean", "psnr_mean", "lpips_mean", "l2_norm_mean", "linf_norm_mean")}
        if "images" in record:
            raw = load_array(record["images"]["raw"], base_dir)
            cooked = load_array(record["images"]["cooked"], base_dir)
            full_quality = compute_image_quality_metrics(raw, cooked, compute_lpips=bool(record.get("compute_lpips", True)))
            quality.update({name: full_quality[name] for name in quality})
        elif record["metadata"].get("modality", "").lower() not in {"tabular", "table"}:
            logger.warning("No image artifacts for %s epsilon=%s; quality columns are NaN.", record["metadata"]["dataset"], record["metadata"]["epsilon"])
        for metric, result in cp_pp.items():
            aliases = {name[:-5] if name.endswith("_mean") else name: value for name, value in quality.items()}
            rows.append({**metadata(record, setting="epsilon_ablation"), "metric": metric, **result, **quality, **aliases})
    output = config.get("output", "results/ablations/epsilon_ablation.csv")
    logger.info("Wrote %s", write_rows(rows, output))


if __name__ == "__main__":
    main()
