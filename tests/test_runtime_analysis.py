from __future__ import annotations

import pytest

from run_runtime_analysis import LATENCY_UNIT, _append_macro_average


def _row(dataset: str, latency: float, samples: int) -> dict:
    return {
        "dataset": dataset,
        "modality": "3d_volume",
        "task_type": "multiclass",
        "method": "datacook",
        "surrogate_model": "resnet18_3d",
        "victim_model": "resnet18_3d",
        "seed": 0,
        "epsilon": 32 / 255,
        "setting": "runtime",
        "attack_type": "",
        "attack_strength": "",
        "resolution": "28x28x28",
        "num_samples": samples,
        "batch_size": 16,
        "device": "cuda",
        "cooking_time_total_seconds": latency * samples / 1_000_000,
        "cooking_time_per_sample": latency,
        "raw_inference_time_per_sample": latency * 10,
        "cooked_inference_time_per_sample": latency * 11,
        "authorized_cooking_latency_per_sample": latency,
        "overhead_ratio": 1.2,
        "gpu_memory_mb": latency * 2,
        "latency_unit": LATENCY_UNIT,
        "aggregation": "dataset",
        "repeats": 5,
        "warmup": 1,
    }


def test_runtime_macro_average_is_equal_weighted_across_datasets() -> None:
    rows = [_row("small", 10.0, 10), _row("large", 30.0, 100)]

    output = _append_macro_average(rows)

    assert len(output) == 3
    average = output[-1]
    assert average["dataset"] == "Average"
    assert average["aggregation"] == "macro_average"
    assert average["latency_unit"] == LATENCY_UNIT
    assert average["num_samples"] == 110
    assert average["cooking_time_per_sample"] == pytest.approx(20.0)
    assert average["raw_inference_time_per_sample"] == pytest.approx(200.0)
    assert average["cooked_inference_time_per_sample"] == pytest.approx(220.0)
    assert average["gpu_memory_mb"] == pytest.approx(40.0)


def test_single_runtime_record_does_not_add_duplicate_average() -> None:
    rows = [_row("only", 10.0, 10)]
    assert _append_macro_average(rows) == rows
