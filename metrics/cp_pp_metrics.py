"""Generalized copyright/privacy (CP/PP) calculations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def _random_baseline(
    metric: str,
    task_type: str | None,
    num_classes: int | None,
    label_prevalence: Sequence[float] | None,
) -> float:
    name = metric.lower()
    if "auc" in name:
        return 0.5
    task = (task_type or "").replace("-", "").replace("_", "").lower()
    if task == "binary":
        if name in {"acc", "balanced_acc", "sensitivity", "specificity"}:
            return 0.5
    elif task == "multiclass" and num_classes and num_classes > 0:
        if name in {"acc", "balanced_acc", "macro_f1", "macro_precision", "macro_sensitivity"}:
            return 1.0 / num_classes
        if name == "macro_specificity":
            return 1.0 - 1.0 / num_classes
    elif task == "multilabel" and label_prevalence is not None:
        prevalence = np.asarray(label_prevalence, dtype=float)
        if name == "subset_acc":
            return float(np.prod(prevalence**2 + (1.0 - prevalence) ** 2))
        if name in {"micro_auc", "macro_auc", "micro_sensitivity", "macro_sensitivity", "micro_specificity", "macro_specificity"}:
            return 0.5
    return float("nan")


def compute_cp_pp(
    clean_metrics: Mapping[str, float],
    authorized_metrics: Mapping[str, float],
    unauthorized_metrics: Mapping[str, float],
    metric_names: Iterable[str],
    random_guess_metrics: Mapping[str, float] | None = None,
    *,
    task_type: str | None = None,
    num_classes: int | None = None,
    label_prevalence: Sequence[float] | None = None,
) -> dict[str, dict[str, float]]:
    """Calculate PP, CP and residual utility for any common metric set.

    PP is ``authorized_cooked - clean`` and CP is ``unauthorized_raw - clean``.
    Negative CP therefore represents utility degradation, preserving the sign
    convention used by the original DataCook experiments.
    """
    output: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        missing = [
            name
            for name, values in (
                ("clean_metrics", clean_metrics),
                ("authorized_metrics", authorized_metrics),
                ("unauthorized_metrics", unauthorized_metrics),
            )
            if metric not in values
        ]
        if missing:
            raise KeyError(f"Metric {metric!r} is missing from {', '.join(missing)}.")
        clean = float(clean_metrics[metric])
        authorized = float(authorized_metrics[metric])
        unauthorized = float(unauthorized_metrics[metric])
        if random_guess_metrics is not None and metric in random_guess_metrics:
            random_guess = float(random_guess_metrics[metric])
        else:
            random_guess = _random_baseline(metric, task_type, num_classes, label_prevalence)
        output[metric] = {
            "clean": clean,
            "authorized_cooked": authorized,
            "unauthorized_raw": unauthorized,
            "cp": unauthorized - clean,
            "pp": authorized - clean,
            "random_guess": random_guess,
            "residual_utility": unauthorized - random_guess if np.isfinite(random_guess) else float("nan"),
        }
    return output
