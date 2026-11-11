import pandas as pd
import pytest

from analysis.merge_result_shards import merge_shards


def _row(seed, value):
    return {
        "dataset": "Toy", "method": "datacook", "seed": seed, "epsilon": 8 / 255,
        "setting": "adaptive_attack", "attack_type": "finetune",
        "attack_strength": "epochs=10", "metric": "acc",
        "attacked_unauthorized_raw": value,
    }


def test_merge_shards_combines_non_overlapping_rows(tmp_path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    pd.DataFrame([_row(0, 0.6)]).to_csv(first, index=False)
    pd.DataFrame([_row(1, 0.7)]).to_csv(second, index=False)
    result = merge_shards([first, second])
    assert result["seed"].tolist() == [0, 1]


def test_merge_shards_rejects_conflicting_identity(tmp_path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    pd.DataFrame([_row(0, 0.6)]).to_csv(first, index=False)
    pd.DataFrame([_row(0, 0.9)]).to_csv(second, index=False)
    with pytest.raises(ValueError, match="Conflicting result rows"):
        merge_shards([first, second])
