"""Unified binary, multiclass and multilabel classification metrics.

The public functions accept NumPy arrays, Python sequences, or CPU/GPU tensors.
Undefined AUC values (for example, a test split containing only one class) are
reported as ``nan`` instead of aborting a long experiment.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


_TASK_ALIASES = {
    "binary-class": "binary",
    "binary_class": "binary",
    "multi-class": "multiclass",
    "multi_class": "multiclass",
    "multilabel": "multilabel",
    "multi-label": "multilabel",
    "multi_label": "multilabel",
}


def _numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray, **kwargs: Any) -> float:
    try:
        return float(roc_auc_score(y_true, y_score, **kwargs))
    except ValueError as exc:
        warnings.warn(f"AUC is undefined for this split: {exc}", RuntimeWarning)
        return float("nan")


def _as_binary_probability(y_prob: np.ndarray) -> np.ndarray:
    if y_prob.ndim == 1:
        return y_prob.astype(float)
    if y_prob.ndim == 2 and y_prob.shape[1] == 1:
        return y_prob[:, 0].astype(float)
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        if not np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError("Two-column binary y_prob rows must sum to 1; pass softmax probabilities, not logits.")
        return y_prob[:, 1].astype(float)
    raise ValueError("Binary y_prob must have shape (N,), (N, 1), or (N, 2).")


def _validate_probabilities(y_prob: np.ndarray, name: str = "y_prob") -> None:
    if not np.all(np.isfinite(y_prob)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError(f"{name} must contain probabilities in [0, 1], not logits.")


def _multiclass_auc(
    y_true: np.ndarray, y_prob: np.ndarray, average: str
) -> float:
    """OvR AUC that tolerates classes missing from a test split."""
    aucs, weights = [], []
    for class_index in range(y_prob.shape[1]):
        binary_true = (y_true == class_index).astype(int)
        if np.unique(binary_true).size < 2:
            continue
        aucs.append(roc_auc_score(binary_true, y_prob[:, class_index]))
        weights.append(float(binary_true.sum()))
    if not aucs:
        warnings.warn("AUC is undefined because no class has both outcomes.", RuntimeWarning)
        return float("nan")
    if average == "macro":
        return float(np.mean(aucs))
    return float(np.average(aucs, weights=weights))


def _multiclass_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> tuple[float, float]:
    values, weights = [], []
    for class_index in range(num_classes):
        positive = y_true == class_index
        negative = ~positive
        tn = np.logical_and(negative, y_pred != class_index).sum()
        fp = np.logical_and(negative, y_pred == class_index).sum()
        values.append(_safe_divide(float(tn), float(tn + fp)))
        weights.append(float(positive.sum()))
    macro = float(np.mean(values))
    weighted = float(np.average(values, weights=weights)) if sum(weights) else macro
    return macro, weighted


def compute_ece(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 15,
    task_type: str | None = None,
) -> float:
    """Compute equal-width expected calibration error.

    Binary and multiclass use top-label confidence. Multilabel ECE treats every
    label decision as a binary prediction, which makes the returned value easy
    to compare across datasets with different label counts.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    truth = _numpy(y_true)
    prob = _numpy(y_prob)
    if truth is None or prob is None:
        raise ValueError("y_true and y_prob are required for ECE.")
    task = _TASK_ALIASES.get(str(task_type).lower(), str(task_type).lower()) if task_type else None
    _validate_probabilities(prob)

    if task == "multilabel" or (task is None and truth.ndim == 2 and prob.ndim == 2 and prob.shape[1] != 2):
        if truth.shape != prob.shape:
            raise ValueError("Multilabel y_true and y_prob must have the same shape.")
        predicted = prob >= 0.5
        confidence = np.where(predicted, prob, 1.0 - prob).reshape(-1)
        correct = (predicted == truth.astype(bool)).astype(float).reshape(-1)
    elif prob.ndim == 2 and prob.shape[1] > 1 and task != "binary":
        if prob.shape[0] != truth.reshape(-1).shape[0]:
            raise ValueError("y_true and y_prob contain different sample counts.")
        predicted = np.argmax(prob, axis=1)
        confidence = np.max(prob, axis=1)
        correct = (predicted == truth.reshape(-1).astype(int)).astype(float)
    else:
        binary_prob = _as_binary_probability(prob)
        binary_truth = truth.reshape(-1).astype(int)
        if binary_prob.shape[0] != binary_truth.shape[0]:
            raise ValueError("y_true and y_prob contain different sample counts.")
        predicted = binary_prob >= 0.5
        confidence = np.where(predicted, binary_prob, 1.0 - binary_prob)
        correct = (predicted == binary_truth.astype(bool)).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-inclusive final bin prevents confidence == 1 from being discarded.
    bin_ids = np.minimum(np.digitize(confidence, edges[1:-1], right=False), n_bins - 1)
    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any = None,
    y_prob: Any = None,
    task_type: str = "multiclass",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute task-appropriate classification metrics.

    ``y_prob`` must contain probabilities. Passing logits is deliberately an
    error so that softmax/sigmoid mistakes cannot silently corrupt AUC or ECE.
    Predictions are inferred from probabilities when ``y_pred`` is omitted.
    """
    task = _TASK_ALIASES.get(task_type.lower(), task_type.lower())
    if task not in {"binary", "multiclass", "multilabel"}:
        raise ValueError(f"Unsupported task_type {task_type!r}.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")

    truth = _numpy(y_true)
    pred = _numpy(y_pred)
    prob = _numpy(y_prob)
    if truth is None or truth.size == 0:
        raise ValueError("y_true must contain at least one sample.")
    if prob is not None:
        _validate_probabilities(prob)

    if task == "binary":
        truth = truth.reshape(-1).astype(int)
        if not set(np.unique(truth)).issubset({0, 1}):
            raise ValueError("Binary y_true must contain only 0 and 1.")
        binary_prob = _as_binary_probability(prob) if prob is not None else None
        if pred is None:
            if binary_prob is None:
                raise ValueError("Either y_pred or y_prob is required.")
            pred = (binary_prob >= threshold).astype(int)
        pred = pred.reshape(-1).astype(int)
        if pred.shape != truth.shape:
            raise ValueError("y_true and y_pred contain different sample counts.")
        tn = np.logical_and(truth == 0, pred == 0).sum()
        fp = np.logical_and(truth == 0, pred == 1).sum()
        fn = np.logical_and(truth == 1, pred == 0).sum()
        tp = np.logical_and(truth == 1, pred == 1).sum()
        return {
            "acc": float(accuracy_score(truth, pred)),
            "balanced_acc": float(balanced_accuracy_score(truth, pred)),
            "auc": _safe_auc(truth, binary_prob) if binary_prob is not None else float("nan"),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(truth, pred, average="weighted", zero_division=0)),
            "precision": float(precision_score(truth, pred, zero_division=0)),
            "sensitivity": _safe_divide(float(tp), float(tp + fn)),
            "specificity": _safe_divide(float(tn), float(tn + fp)),
            "ece": compute_ece(truth, binary_prob, task_type=task) if binary_prob is not None else float("nan"),
        }

    if task == "multiclass":
        truth = truth.reshape(-1).astype(int)
        if pred is None:
            if prob is None:
                raise ValueError("Either y_pred or y_prob is required.")
            if prob.ndim != 2:
                raise ValueError("Multiclass y_prob must have shape (N, C).")
            pred = np.argmax(prob, axis=1)
        pred = pred.reshape(-1).astype(int)
        if pred.shape != truth.shape:
            raise ValueError("y_true and y_pred contain different sample counts.")
        if prob is not None and (prob.ndim != 2 or prob.shape[0] != truth.shape[0]):
            raise ValueError("Multiclass y_prob must have shape (N, C).")
        if prob is not None and not np.allclose(prob.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError("Multiclass y_prob rows must sum to 1; pass softmax probabilities, not logits.")
        num_classes = prob.shape[1] if prob is not None else int(max(truth.max(), pred.max()) + 1)
        macro_specificity, _ = _multiclass_specificity(truth, pred, num_classes)
        return {
            "acc": float(accuracy_score(truth, pred)),
            "balanced_acc": float(balanced_accuracy_score(truth, pred)),
            "macro_auc": _multiclass_auc(truth, prob, "macro") if prob is not None else float("nan"),
            "weighted_auc": _multiclass_auc(truth, prob, "weighted") if prob is not None else float("nan"),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(truth, pred, average="weighted", zero_division=0)),
            "macro_precision": float(precision_score(truth, pred, average="macro", zero_division=0)),
            "macro_sensitivity": float(recall_score(truth, pred, average="macro", zero_division=0)),
            "macro_specificity": macro_specificity,
            "ece": compute_ece(truth, prob, task_type=task) if prob is not None else float("nan"),
        }

    truth = np.asarray(truth).astype(int)
    if truth.ndim != 2:
        raise ValueError("Multilabel y_true must have shape (N, L).")
    if pred is None:
        if prob is None:
            raise ValueError("Either y_pred or y_prob is required.")
        pred = (prob >= threshold).astype(int)
    pred = np.asarray(pred).astype(int)
    if pred.shape != truth.shape:
        raise ValueError("Multilabel y_true and y_pred must have the same shape.")
    if prob is not None and prob.shape != truth.shape:
        raise ValueError("Multilabel y_true and y_prob must have the same shape.")

    tn = np.logical_and(truth == 0, pred == 0).sum(axis=0)
    fp = np.logical_and(truth == 0, pred == 1).sum(axis=0)
    macro_specificity = float(np.mean([_safe_divide(float(a), float(a + b)) for a, b in zip(tn, fp)]))
    micro_specificity = _safe_divide(float(tn.sum()), float((tn + fp).sum()))
    macro_auc = float(np.nanmean([
        _safe_auc(truth[:, i], prob[:, i]) for i in range(truth.shape[1])
    ])) if prob is not None else float("nan")
    return {
        "subset_acc": float(accuracy_score(truth, pred)),
        "hamming_loss": float(hamming_loss(truth, pred)),
        "micro_auc": _safe_auc(truth.reshape(-1), prob.reshape(-1)) if prob is not None else float("nan"),
        "macro_auc": macro_auc,
        "micro_f1": float(f1_score(truth, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(truth, pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(truth, pred, average="macro", zero_division=0)),
        "micro_sensitivity": float(recall_score(truth, pred, average="micro", zero_division=0)),
        "macro_sensitivity": float(recall_score(truth, pred, average="macro", zero_division=0)),
        "micro_specificity": micro_specificity,
        "macro_specificity": macro_specificity,
        "ece": compute_ece(truth, prob, task_type=task) if prob is not None else float("nan"),
    }
