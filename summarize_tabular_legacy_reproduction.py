#!/usr/bin/env python3
"""Compare strict legacy tabular reproduction results with the old table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGETS = {
    "Diabetes": {"metric": "acc", "cp": -18.18, "pp": 1.31},
    "HepatitisC": {"metric": "acc", "cp": -31.71, "pp": 0.00},
    "SystemicDiseases": {"metric": "subset_acc", "cp": -50.22, "pp": -0.45},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        default="results/revision_pipeline_tabular_legacy_exact",
    )
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.result_root).expanduser().resolve()
    paths = sorted((root / "datasets").glob("*/epsilon_ablation.csv"))
    if not paths:
        raise FileNotFoundError(f"No epsilon_ablation.csv files found under {root}")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)

    rows = []
    for dataset, target in TARGETS.items():
        selected = frame[
            (frame["dataset"] == dataset)
            & (frame["seed"] == 42)
            & (frame["epsilon"] == 5.0)
            & (frame["metric"] == target["metric"])
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Expected exactly one seed=42, epsilon=5 row for {dataset}/"
                f"{target['metric']}, found {len(selected)}."
            )
        row = selected.iloc[0]
        observed_cp = 100.0 * float(row["cp"])
        observed_pp = 100.0 * float(row["pp"])
        cp_error = observed_cp - target["cp"]
        pp_error = observed_pp - target["pp"]
        rows.append({
            "dataset": dataset,
            "metric": target["metric"],
            "clean": 100.0 * float(row["clean"]),
            "authorized_cooked": 100.0 * float(row["authorized_cooked"]),
            "unauthorized_raw": 100.0 * float(row["unauthorized_raw"]),
            "observed_cp": observed_cp,
            "target_cp": target["cp"],
            "cp_error": cp_error,
            "observed_pp": observed_pp,
            "target_pp": target["pp"],
            "pp_error": pp_error,
            "within_tolerance": (
                abs(cp_error) <= args.tolerance
                and abs(pp_error) <= args.tolerance
            ),
        })

    output = pd.DataFrame(rows)
    output_path = root / "target_reproduction.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(output.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"Wrote {output_path}")
    if args.strict and not bool(output["within_tolerance"].all()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
