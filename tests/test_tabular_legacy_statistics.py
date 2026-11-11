import pandas as pd

from analysis.tabular_legacy_statistics import build_tests


def test_tabular_legacy_statistics_pairs_seeds_and_applies_holm():
    rows = []
    for epsilon in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        for seed in range(8):
            rows.append({
                "dataset": "Diabetes",
                "task_type": "binary",
                "epsilon": epsilon,
                "metric": "acc",
                "seed": seed,
                "clean": 0.8,
                "authorized_cooked": 0.79,
                "unauthorized_raw": 0.6,
            })
    result = build_tests(pd.DataFrame(rows), {})
    assert len(result) == 12
    assert set(result["num_pairs"]) == {8}
    assert set(result["num_nonzero"]) == {8}
    assert result["p_value_holm"].notna().all()
    assert result["significant_holm"].all()
