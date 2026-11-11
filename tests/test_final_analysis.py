import numpy as np
import pandas as pd

from analysis.finalize_revision_results import (
    build_budget_all,
    build_epsilon_ablation,
    build_significance,
)


def _attack_rows(strength="epochs=10,lr=0.0001"):
    rows = []
    for seed in (0, 1, 2):
        for attack_type, after in (("finetune", 0.7), ("robust_finetune", 0.65)):
            rows.append({
                "dataset": "Toy", "modality": "2d_image", "task_type": "multiclass",
                "method": "datacook", "surrogate_model": "r18", "victim_model": "r18",
                "seed": seed, "epsilon": 8 / 255, "setting": "adaptive_attack",
                "attack_type": attack_type, "attack_strength": strength, "metric": "acc",
                "clean": 0.8, "authorized_cooked": 0.8, "unauthorized_raw": 0.5,
                "attacked_unauthorized_raw": after, "cp_before_attack": -0.3,
                "cp_after_attack": after - 0.8, "recovery_rate": (after - 0.5) / 0.3,
                "recovery_rate_unclipped": (after - 0.5) / 0.3,
            })
    return pd.DataFrame(rows)


def test_epsilon_ablation_merges_quality_by_run():
    cp = pd.DataFrame([
        {"dataset": "Toy", "modality": "2d_image", "task_type": "multiclass",
         "method": "datacook", "surrogate_model": "r18", "victim_model": "r18",
         "seed": 0, "epsilon": 8 / 255, "metric": "acc", "cp": -0.2, "pp": 0.0},
    ])
    quality = pd.DataFrame([
        {"dataset": "Toy", "modality": "2d_image", "task_type": "multiclass",
         "method": "datacook", "surrogate_model": "r18", "victim_model": "r18",
         "seed": 0, "epsilon": 8 / 255, "ssim_mean": 0.95, "ssim_std": 0.01},
    ])
    result = build_epsilon_ablation(cp, quality)
    assert result.loc[0, "epsilon_pixel"] == 8
    assert result.loc[0, "ssim_mean"] == 0.95


def test_budget_all_reuses_full_and_adds_fixed_budget():
    full = _attack_rows()
    budget = _attack_rows("epochs=10,lr=0.0001,n=25")
    budget["raw_subset_size"] = 25
    result = build_budget_all(full, budget, {"Toy": 60})
    assert set(result["raw_subset_size"]) == {25, 60}
    assert set(result["raw_subset_budget"]) == {"25", "full"}
    assert len(result) == 12


def test_significance_reports_effect_and_holm_value():
    attacks = _attack_rows()
    cp = pd.DataFrame([
        {"dataset": "Toy", "epsilon": 8 / 255, "metric": "acc", "seed": seed,
         "clean": 0.8, "unauthorized_raw": 0.5}
        for seed in (0, 1, 2)
    ])
    budget_all = build_budget_all(attacks, pd.DataFrame(), {"Toy": 60})
    result = build_significance(cp, attacks, budget_all)
    assert not result.empty
    assert result["p_value"].notna().all()
    assert result["p_value_holm"].between(0, 1).all()
    before_after = result[result["comparison"].eq("attack_before_vs_after")]
    assert np.all(before_after["mean_difference_b_minus_a"] > 0)


def test_summary_preserves_identical_infinite_psnr():
    from analysis.finalize_revision_results import _summary_table

    frame = pd.DataFrame({"dataset": ["Breast", "Breast", "Breast"], "psnr_mean": [np.inf] * 3})
    result = _summary_table(frame, ["dataset"], ["psnr_mean"])
    assert np.isposinf(result.loc[0, "psnr_mean_mean"])
    assert result.loc[0, "psnr_mean_std"] == 0.0
