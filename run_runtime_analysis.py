#!/usr/bin/env python3
"""Task 7: robust cooking and inference latency measurements."""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from revision_experiments.common import load_array, load_config, metadata, require_records, setup_logging, write_rows
from revision_experiments.modeling import build_cooker, load_model


SECONDS_TO_MICROSECONDS = 1_000_000.0
LATENCY_UNIT = "microseconds_per_sample"
PER_SAMPLE_LATENCY_COLUMNS = (
    "cooking_time_per_sample",
    "raw_inference_time_per_sample",
    "cooked_inference_time_per_sample",
    "authorized_cooking_latency_per_sample",
)
MACRO_AVERAGE_COLUMNS = (*PER_SAMPLE_LATENCY_COLUMNS, "overhead_ratio", "gpu_memory_mb")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_inference(model, inputs: torch.Tensor, batch_size: int, device: torch.device, repeats: int, warmup: int) -> float:
    loader = DataLoader(TensorDataset(inputs), batch_size=batch_size, shuffle=False)
    model.eval()
    elapsed = []
    with torch.no_grad():
        for run in range(warmup + repeats):
            _sync(device)
            start = time.perf_counter()
            for (batch,) in loader:
                model(batch.to(device, non_blocking=True))
            _sync(device)
            if run >= warmup:
                elapsed.append(time.perf_counter() - start)
    return float(np.median(elapsed)) / len(inputs)


def _time_cooking(cooker, inputs: torch.Tensor, device: torch.device, repeats: int, warmup: int):
    elapsed, output = [], None
    batch = inputs.to(device)
    with torch.no_grad():
        for run in range(warmup + repeats):
            _sync(device)
            start = time.perf_counter()
            output = cooker(batch)
            _sync(device)
            if run >= warmup:
                elapsed.append(time.perf_counter() - start)
    if not torch.is_tensor(output):
        output = torch.as_tensor(output, device=device)
    return float(np.median(elapsed)), output.detach().cpu()


def _shared_value(rows: list[Mapping[str, Any]], key: str, fallback: Any = "mixed") -> Any:
    """Return a shared metadata value, or ``fallback`` when rows differ."""
    values = [row.get(key, "") for row in rows]
    return values[0] if values and all(value == values[0] for value in values) else fallback


def _append_macro_average(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append an equal-dataset macro average to a multi-dataset runtime table."""
    if len(rows) < 2:
        return rows

    average = {
        "dataset": "Average",
        "modality": _shared_value(rows, "modality"),
        "task_type": _shared_value(rows, "task_type"),
        "method": _shared_value(rows, "method"),
        "surrogate_model": _shared_value(rows, "surrogate_model"),
        "victim_model": _shared_value(rows, "victim_model"),
        "seed": _shared_value(rows, "seed"),
        "epsilon": _shared_value(rows, "epsilon"),
        "setting": "runtime",
        "attack_type": "",
        "attack_strength": "",
        "resolution": _shared_value(rows, "resolution"),
        "num_samples": sum(int(row["num_samples"]) for row in rows),
        "batch_size": _shared_value(rows, "batch_size"),
        "device": _shared_value(rows, "device"),
        "cooking_time_total_seconds": sum(float(row["cooking_time_total_seconds"]) for row in rows),
        "latency_unit": LATENCY_UNIT,
        "aggregation": "macro_average",
        "repeats": _shared_value(rows, "repeats"),
        "warmup": _shared_value(rows, "warmup"),
    }
    for column in MACRO_AVERAGE_COLUMNS:
        average[column] = float(np.mean([float(row[column]) for row in rows]))
    return [*rows, average]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("runtime-analysis", args.verbose)
    config, base_dir = load_config(args.config)
    device = torch.device(args.device)
    rows = []
    for record in require_records(config):
        inputs = torch.as_tensor(load_array(record["data"]["raw_inputs"], base_dir)).float()
        if len(inputs) == 0:
            raise ValueError("Runtime analysis requires at least one sample.")
        batch_size = int(record.get("batch_size", 128))
        repeats, warmup = int(record.get("repeats", 5)), int(record.get("warmup", 1))
        cooker = build_cooker(record["cooker"], base_dir)
        model = load_model(record["model"], base_dir, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        cooking_total, generated_cooked = _time_cooking(cooker, inputs, device, repeats, warmup)
        if "cooked_inputs" in record["data"]:
            cooked = torch.as_tensor(load_array(record["data"]["cooked_inputs"], base_dir)).float()
        else:
            cooked = generated_cooked
        if cooked.shape != inputs.shape:
            raise ValueError("Cooked and raw inputs have different shapes in runtime analysis.")
        raw_latency = _time_inference(model, inputs, batch_size, device, repeats, warmup)
        cooked_latency = _time_inference(model, cooked, batch_size, device, repeats, warmup)
        cooking_latency_seconds = cooking_total / len(inputs)
        overhead = (
            (cooking_latency_seconds + cooked_latency) / raw_latency
            if raw_latency > 0 else float("nan")
        )
        memory = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else float("nan")
        shape = "x".join(str(value) for value in inputs.shape[2:]) if inputs.ndim >= 4 else str(inputs.shape[-1])
        rows.append({
            **metadata(record, setting="runtime"), "resolution": record.get("resolution", shape),
            "num_samples": len(inputs), "batch_size": batch_size, "device": str(device),
            "cooking_time_total_seconds": cooking_total,
            "cooking_time_per_sample": cooking_latency_seconds * SECONDS_TO_MICROSECONDS,
            "raw_inference_time_per_sample": raw_latency * SECONDS_TO_MICROSECONDS,
            "cooked_inference_time_per_sample": cooked_latency * SECONDS_TO_MICROSECONDS,
            "authorized_cooking_latency_per_sample": cooking_latency_seconds * SECONDS_TO_MICROSECONDS,
            "overhead_ratio": overhead, "gpu_memory_mb": memory,
            "latency_unit": LATENCY_UNIT, "aggregation": "dataset",
            "repeats": repeats, "warmup": warmup,
        })
        logger.info(
            "Timed %s: %.3f microseconds/sample cooking",
            record["metadata"]["dataset"],
            cooking_latency_seconds * SECONDS_TO_MICROSECONDS,
        )
    if bool(config.get("include_average", True)):
        rows = _append_macro_average(rows)
    output = config.get("output", "results/runtime/runtime_summary.csv")
    logger.info("Wrote %s", write_rows(rows, output))


if __name__ == "__main__":
    main()
