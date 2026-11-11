import numpy as np
import pandas as pd

from attacks.jpeg import jpeg_compress
from attacks.smoothing import gaussian_smooth
from attacks.denoise import median_filter
from attacks.augmentation import strong_augment
from run_adaptive_attacks import _merge_results, _select_raw_subset


def test_image_attacks_preserve_shape_and_range():
    images = np.random.default_rng(0).random((2, 3, 12, 12), dtype=np.float32)
    for attacked in (
        jpeg_compress(images, quality=75),
        gaussian_smooth(images, sigma=1.0, kernel_size=3),
        median_filter(images, kernel_size=3),
        strong_augment(images, strength=2, seed=0),
    ):
        assert attacked.shape == images.shape
        assert attacked.min() >= 0
        assert attacked.max() <= 1


def test_merge_results_preserves_inputs_and_deduplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    old = pd.DataFrame([{"attack_type": "jpeg", "metric": "accuracy", "value": 0.5}])
    new = pd.DataFrame([
        {"attack_type": "finetune", "metric": "accuracy", "value": 0.7},
        {"attack_type": "finetune", "metric": "accuracy", "value": 0.7},
    ])
    old.to_csv("old.csv", index=False)
    new.to_csv("new.csv", index=False)

    output = _merge_results(tmp_path / "new.csv", "old.csv", "all.csv")

    combined = pd.read_csv(output)
    assert combined["attack_type"].tolist() == ["jpeg", "finetune"]
    assert len(pd.read_csv("old.csv")) == 1
    assert len(pd.read_csv("new.csv")) == 2


def test_raw_budget_subset_is_reproducible_and_stratified():
    inputs = np.arange(60).reshape(20, 3)
    labels = np.repeat(np.arange(4), 5)
    first_x, first_y = _select_raw_subset(inputs, labels, subset_size=12, seed=7)
    second_x, second_y = _select_raw_subset(inputs, labels, subset_size=12, seed=7)

    assert np.array_equal(first_x, second_x)
    assert np.array_equal(first_y, second_y)
    assert len(first_x) == 12
    assert set(first_y) == {0, 1, 2, 3}
