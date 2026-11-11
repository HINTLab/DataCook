#!/usr/bin/env python3
"""Task 5: sample-level pseudo-label export and quality/CP/PP analysis."""

from __future__ import annotations

import argparse
import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr

from metrics import compute_classification_metrics, compute_cp_pp
from revision_experiments.common import json_value, load_config, load_prediction_artifact, metadata, require_records, resolve_path, setup_logging, write_rows


def _load_pseudo(spec, base_dir, task_type: str, threshold: float):
    path = resolve_path(spec["path"] if isinstance(spec, dict) else spec, base_dir)
    if not path.is_file() or path.suffix.lower() != ".npz":
        raise FileNotFoundError(f"Pseudo-label artifact must be an existing .npz file: {path}")
    keys = spec.get("keys", {}) if isinstance(spec, dict) else {}
    with np.load(path, allow_pickle=False) as archive:
        true = np.asarray(archive[keys.get("y_true", "y_true")])
        sample_ids = np.asarray(archive[keys.get("sample_id", "sample_id")]) if keys.get("sample_id", "sample_id") in archive else np.arange(len(true))
        prob_key, logits_key = keys.get("y_prob", "y_prob"), keys.get("logits", "logits")
        if prob_key in archive:
            prob = np.asarray(archive[prob_key], dtype=float)
        elif logits_key in archive:
            logits = np.asarray(archive[logits_key], dtype=float)
            task = task_type.replace("-", "").replace("_", "").lower()
            if task == "multilabel" or (task == "binary" and (logits.ndim == 1 or logits.shape[-1] == 1)):
                prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
            else:
                shifted = logits - logits.max(axis=-1, keepdims=True)
                exp = np.exp(shifted)
                prob = exp / exp.sum(axis=-1, keepdims=True)
        else:
            raise KeyError(f"{path} must contain probabilities or logits.")

    if not np.all(np.isfinite(prob)) or np.any(prob < 0) or np.any(prob > 1):
        raise ValueError(f"Pseudo-label probabilities in {path} must be finite and in [0,1].")
    task = task_type.replace("-", "").replace("_", "").lower()
    if task == "binary":
        if prob.ndim == 2 and prob.shape[1] == 2:
            prob = prob[:, 1]
        prob = prob.reshape(-1)
        pseudo = (prob >= threshold).astype(int)
        confidence = np.maximum(prob, 1.0 - prob)
        true = true.reshape(-1).astype(int)
        correct = pseudo == true
    elif task == "multiclass":
        pseudo = np.argmax(prob, axis=1)
        confidence = np.max(prob, axis=1)
        true = true.reshape(-1).astype(int)
        correct = pseudo == true
    elif task == "multilabel":
        pseudo = (prob >= threshold).astype(int)
        confidence = np.mean(np.maximum(prob, 1.0 - prob), axis=1)
        true = np.asarray(true).astype(int)
        if true.shape != pseudo.shape:
            raise ValueError("Multilabel pseudo y_true and probabilities must have the same shape.")
        correct = np.all(pseudo == true, axis=1)
    else:
        raise ValueError(f"Unsupported task_type {task_type!r}.")
    return sample_ids, true, pseudo, confidence, correct


def _by_class(true: np.ndarray, pseudo: np.ndarray, task_type: str) -> dict[str, float]:
    task = task_type.replace("-", "").replace("_", "").lower()
    output = {}
    if task == "multilabel":
        for index in range(true.shape[1]):
            output[str(index)] = float(np.mean(true[:, index] == pseudo[:, index]))
    else:
        for label in np.unique(true):
            mask = true == label
            output[str(int(label))] = float(np.mean(pseudo[mask] == true[mask]))
    return output


def _cp_pp(record, base_dir):
    if "predictions" not in record:
        supplied = record.get("cp_pp", {})
        return supplied.get("cp_acc", np.nan), supplied.get("pp_acc", np.nan), supplied.get("cp_auc", np.nan), supplied.get("pp_auc", np.nan)
    task = record["metadata"]["task_type"]
    metrics = {}
    for setting in ("clean", "authorized_cooked", "unauthorized_raw"):
        artifact = load_prediction_artifact(record["predictions"][setting], base_dir)
        metrics[setting] = compute_classification_metrics(artifact["y_true"], artifact["y_pred"], artifact["y_prob"], task, float(record.get("threshold", 0.5)))
    acc_name = "subset_acc" if task.replace("-", "").replace("_", "").lower() == "multilabel" else "acc"
    auc_name = "auc" if task.replace("-", "").replace("_", "").lower() == "binary" else "macro_auc"
    result = compute_cp_pp(metrics["clean"], metrics["authorized_cooked"], metrics["unauthorized_raw"], [acc_name, auc_name], task_type=task, num_classes=record.get("num_classes"), label_prevalence=record.get("label_prevalence"))
    return result[acc_name]["cp"], result[acc_name]["pp"], result[auc_name]["cp"], result[auc_name]["pp"]


def _correlation(x, y, method):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float((pearsonr if method == "pearson" else spearmanr)(x[mask], y[mask]).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("pseudo-label-analysis", args.verbose)
    config, base_dir = load_config(args.config)
    sample_rows, quality_rows = [], []
    for record in require_records(config):
        task = record["metadata"]["task_type"]
        ids, true, pseudo, confidence, correct = _load_pseudo(record["pseudo_labels"], base_dir, task, float(record.get("threshold", 0.5)))
        meta = metadata(record, setting="pseudo_labels")
        for sample_id, target, predicted, conf, is_correct in zip(ids, true, pseudo, confidence, correct):
            sample_rows.append({
                **meta, "sample_id": sample_id, "true_label": json_value(target),
                "pseudo_label": json_value(predicted), "pseudo_confidence": float(conf),
                "surrogate_correct": bool(is_correct),
            })
        cp_acc, pp_acc, cp_auc, pp_auc = _cp_pp(record, base_dir)
        quality_rows.append({
            **meta, "pseudo_acc": float(np.mean(correct)), "mean_confidence": float(np.mean(confidence)),
            "pseudo_acc_by_class": json_value(_by_class(true, pseudo, task)),
            "cp_acc": cp_acc, "pp_acc": pp_acc, "cp_auc": cp_auc, "pp_auc": pp_auc,
        })

    group_fields = ["dataset", "modality", "task_type", "method", "surrogate_model"]
    for row in quality_rows:
        peers = [candidate for candidate in quality_rows if all(candidate[field] == row[field] for field in group_fields)]
        pseudo_acc = np.asarray([item["pseudo_acc"] for item in peers])
        confidence = np.asarray([item["mean_confidence"] for item in peers])
        for outcome in ("cp_acc", "pp_acc", "cp_auc", "pp_auc"):
            target = np.asarray([item[outcome] for item in peers], dtype=float)
            row[f"pearson_pseudo_acc_{outcome}"] = _correlation(pseudo_acc, target, "pearson")
            row[f"spearman_pseudo_acc_{outcome}"] = _correlation(pseudo_acc, target, "spearman")
            row[f"pearson_confidence_{outcome}"] = _correlation(confidence, target, "pearson")
            row[f"spearman_confidence_{outcome}"] = _correlation(confidence, target, "spearman")
        row["pearson_cp"] = row["pearson_pseudo_acc_cp_acc"]
        row["spearman_cp"] = row["spearman_pseudo_acc_cp_acc"]
        row["pearson_pp"] = row["pearson_pseudo_acc_pp_acc"]
        row["spearman_pp"] = row["spearman_pseudo_acc_pp_acc"]
        if len(peers) < 2:
            warnings.warn(f"Only one run is available for {row['dataset']}; correlations are NaN.", RuntimeWarning)

    outputs = config.get("outputs", {})
    samples_path = write_rows(sample_rows, outputs.get("samples", "results/analysis/pseudo_label_samples.csv"))
    quality_path = write_rows(quality_rows, outputs.get("quality", "results/analysis/pseudo_label_quality.csv"))
    logger.info("Wrote %s and %s", samples_path, quality_path)


if __name__ == "__main__":
    main()
