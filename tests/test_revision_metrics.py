import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from metrics.classification_metrics import compute_classification_metrics
from metrics.cp_pp_metrics import compute_cp_pp
from metrics.image_quality import _lpips_values, compute_image_quality_metrics


def test_binary_metrics_and_ece():
    truth = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.4, 0.6, 0.9])
    values = compute_classification_metrics(truth, y_prob=prob, task_type="binary")
    assert values["acc"] == 1.0
    assert values["auc"] == 1.0
    assert values["sensitivity"] == 1.0
    assert values["specificity"] == 1.0
    assert 0 <= values["ece"] <= 1


def test_multiclass_metrics_with_missing_class_in_truth():
    truth = np.array([0, 0, 1, 1])
    prob = np.array([[.8, .1, .1], [.7, .2, .1], [.1, .8, .1], [.2, .7, .1]])
    values = compute_classification_metrics(truth, y_prob=prob, task_type="multiclass")
    assert values["acc"] == 1.0
    assert values["macro_auc"] == 1.0
    assert values["macro_specificity"] == 1.0


def test_multiclass_rejects_non_probability_scores():
    with pytest.raises(ValueError, match="sum to 1"):
        compute_classification_metrics(
            np.array([0, 1]), y_prob=np.array([[.8, .8], [.2, .2]]), task_type="multiclass"
        )


def test_multilabel_metrics():
    truth = np.array([[1, 0], [0, 1], [1, 1]])
    prob = np.array([[.9, .1], [.2, .8], [.8, .7]])
    values = compute_classification_metrics(truth, y_prob=prob, task_type="multi-label")
    assert values["subset_acc"] == 1.0
    assert values["hamming_loss"] == 0.0
    assert values["micro_f1"] == 1.0


def test_cp_pp_random_baseline():
    clean = {"acc": .8, "auc": .9}
    auth = {"acc": .82, "auc": .91}
    unauth = {"acc": .55, "auc": .6}
    result = compute_cp_pp(clean, auth, unauth, ["acc", "auc"], task_type="binary")
    assert math.isclose(result["acc"]["cp"], -.25)
    assert math.isclose(result["acc"]["pp"], .02)
    assert result["auc"]["random_guess"] == .5


def test_image_quality_identical_without_lpips():
    images = np.zeros((2, 1, 8, 8), dtype=np.float32)
    values = compute_image_quality_metrics(images, images, compute_lpips=False)
    assert values["ssim_mean"] == 1.0
    assert math.isinf(values["psnr_mean"])
    assert values["l2_norm_mean"] == 0.0
    assert values["linf_norm_mean"] == 0.0


def test_psnr_uses_aggregate_mse_when_some_images_are_identical():
    raw = np.zeros((2, 1, 8, 8), dtype=np.float32)
    cooked = raw.copy()
    cooked[1] = 0.1
    values = compute_image_quality_metrics(raw, cooked, compute_lpips=False)
    expected = 10 * math.log10(1.0 / 0.005)
    assert math.isfinite(values["psnr_mean"])
    assert math.isclose(values["psnr_mean"], expected, rel_tol=1e-6)
    assert values["psnr_std"] == 0.0


def test_3d_volume_quality_without_lpips():
    raw = np.zeros((2, 1, 8, 8, 8), dtype=np.float32)
    cooked = raw.copy()
    cooked[1, :, 2:6, 2:6, 2:6] = 0.1
    values = compute_image_quality_metrics(raw, cooked, compute_lpips=False)
    assert 0 < values["ssim_mean"] < 1
    assert math.isfinite(values["psnr_mean"])
    assert values["l2_norm_mean"] > 0
    assert math.isclose(values["linf_norm_mean"], 0.05, rel_tol=1e-6)
    assert math.isnan(values["lpips_mean"])


def test_3d_volume_rejects_lpips():
    volumes = np.zeros((1, 1, 8, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="LPIPS is a 2D image metric"):
        compute_image_quality_metrics(volumes, volumes, compute_lpips=True)


def test_lpips_resizes_medmnist_and_uses_bounded_batches(monkeypatch):
    observed_shapes = []

    class FakeLPIPS:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, raw, cooked):
            observed_shapes.append(tuple(raw.shape))
            return (raw - cooked).square().mean(dim=(1, 2, 3), keepdim=True)

    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=lambda net: FakeLPIPS()))
    raw = np.zeros((130, 3, 28, 28), dtype=np.float32)
    cooked = np.full_like(raw, 0.01)
    values = _lpips_values(raw, cooked, device="cpu", batch_size=64)

    assert len(values) == 130
    assert observed_shapes == [(64, 3, 64, 64), (64, 3, 64, 64), (2, 3, 64, 64)]


def test_lpips_runtime_error_is_not_silently_converted_to_nan(monkeypatch):
    class FailingLPIPS:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, _raw, _cooked):
            raise RuntimeError("forward failed")

    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=lambda net: FailingLPIPS()))
    images = np.zeros((1, 3, 28, 28), dtype=np.float32)
    with pytest.raises(RuntimeError, match="forward failed"):
        _lpips_values(images, images, device="cpu")
