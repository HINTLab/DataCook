#!/usr/bin/env python3
"""Plot task-aware perturbation-budget trends across modalities.

Binary tasks use AUC and multiclass/multilabel tasks use macro-AUC. CP and PP
are plotted as signed percentage-point changes. Thin lines show the median
within each dataset, while thick lines and bands show the median and IQR across
dataset-level medians. Collapsed and single-budget settings are excluded.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Avoid a non-writable user-level Matplotlib cache on managed machines.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/datacook-matplotlib")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


DEFAULT_2D = [
    Path("results/revision_pipeline/summary/cp_pp_summary.csv"),
    Path("results/revision_pipeline_extra2d/summary/cp_pp_summary.csv"),
]
DEFAULT_3D = Path("results/revision_pipeline_3d/summary/cp_pp_summary.csv")
DEFAULT_TABULAR = Path(
    "downloaded_tabular_legacy_extended/results/"
    "revision_pipeline_tabular_legacy_extended/summary/cp_pp_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path("results/revision_pipeline/final_figures")

MODALITY_ORDER = ["2D", "3D", "Tabular"]
METRIC_LABELS = {
    "cp": r"CP (unauthorized $-$ clean; $\downarrow$)",
    "pp": r"PP (authorized $-$ clean; $\uparrow$)",
}
COLORS = {
    "cp": "#0072B2",  # Okabe-Ito blue
    "pp": "#D55E00",  # Okabe-Ito vermillion
}


def _read_many(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing budget-trend input: {resolved}")
        frames.append(pd.read_csv(resolved))
    return pd.concat(frames, ignore_index=True)


def _primary_rows(frame: pd.DataFrame, modality: str) -> pd.DataFrame:
    task_type = (
        frame["task_type"].astype(str).str.lower()
        .str.replace("-", "", regex=False)
        .str.replace("_", "", regex=False)
    )
    primary_metric = task_type.map({
        "binary": "auc",
        "multiclass": "macro_auc",
        "multilabel": "macro_auc",
    })
    selected = frame[frame["metric"].eq(primary_metric)].copy()

    if modality in {"2D", "3D"}:
        selected["budget"] = (pd.to_numeric(selected["epsilon"]) * 255.0).round(6)
        selected["budget_unit"] = "pixel"
    else:
        selected["budget"] = pd.to_numeric(selected["epsilon"])
        selected["budget_unit"] = "standardized_feature_linf"

    if modality == "3D":
        selected = selected[~selected["dataset"].eq("SynapseMNIST3D")].copy()

    budget_counts = selected.groupby("dataset")["budget"].transform("nunique")
    selected = selected[budget_counts.ge(2)].copy()

    if selected.empty:
        raise ValueError(f"No primary metric rows found for {modality}.")
    for column in ["clean", "authorized_cooked", "unauthorized_raw"]:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    selected["modality_group"] = modality
    selected["cp"] = (
        selected["unauthorized_raw"] - selected["clean"]
    ) * 100.0
    selected["pp"] = (
        selected["authorized_cooked"] - selected["clean"]
    ) * 100.0
    return selected


def load_budget_data(
    paths_2d: list[Path], path_3d: Path, path_tabular: Path
) -> pd.DataFrame:
    frames = [
        _primary_rows(_read_many(paths_2d), "2D"),
        _primary_rows(_read_many([path_tabular]), "Tabular"),
        _primary_rows(_read_many([path_3d]), "3D"),
    ]
    return pd.concat(frames, ignore_index=True)


def summarize_budget_data(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for modality in MODALITY_ORDER:
        selected = frame[frame["modality_group"].eq(modality)]
        for metric in METRIC_LABELS:
            dataset_summaries: list[dict[str, object]] = []
            for (dataset, budget), group in selected.groupby(["dataset", "budget"]):
                values = group[metric].dropna()
                record = {
                    "modality": modality,
                    "dataset": dataset,
                    "budget": budget,
                    "budget_unit": group["budget_unit"].iloc[0],
                    "metric": metric,
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "num_observations": len(values),
                }
                rows.append(record)
                dataset_summaries.append(record)

            dataset_frame = pd.DataFrame(dataset_summaries)
            for budget, group in dataset_frame.groupby("budget"):
                values = group["median"].dropna()
                rows.append({
                    "modality": modality,
                    "dataset": "Across-dataset summary",
                    "budget": budget,
                    "budget_unit": group["budget_unit"].iloc[0],
                    "metric": metric,
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "num_observations": len(values),
                })
    return pd.DataFrame(rows).sort_values(
        ["metric", "modality", "dataset", "budget"]
    ).reset_index(drop=True)


def _plot_panel(ax: plt.Axes, summary: pd.DataFrame, modality: str, metric: str) -> None:
    selected = summary[
        summary["modality"].eq(modality) & summary["metric"].eq(metric)
    ]
    pooled = selected[selected["dataset"].eq("Across-dataset summary")].sort_values("budget")
    individual = selected[~selected["dataset"].eq("Across-dataset summary")]
    color = COLORS[metric]
    budgets = pooled["budget"].to_numpy(dtype=float)
    positions = np.arange(len(budgets), dtype=float)
    position_by_budget = dict(zip(budgets, positions))

    for dataset, dataset_rows in individual.groupby("dataset"):
        dataset_rows = dataset_rows.sort_values("budget")
        dataset_positions = [position_by_budget[float(value)] for value in dataset_rows["budget"]]
        ax.plot(
            dataset_positions, dataset_rows["median"],
            color="#A6A6A6", linewidth=0.75, alpha=0.75,
            marker="o", markersize=1.8, markeredgewidth=0, zorder=1,
        )
    ax.plot(
        positions, pooled["median"],
        color=color, linewidth=1.8,
        linestyle="-" if metric == "cp" else "--",
        marker="o" if metric == "cp" else "s",
        markersize=4.0, markeredgecolor="white", markeredgewidth=0.4,
        zorder=3,
    )
    ax.axhline(
        0.0, color="#777777", linewidth=0.7, linestyle=(0, (3, 2)),
        alpha=0.8, zorder=0,
    )
    labels = [f"{value:g}" for value in budgets]
    ax.set_xticks(positions, labels)
    if modality == "Tabular":
        ax.set_xlabel(r"$\epsilon$ (standardized-feature $L_\infty$)")
    else:
        ax.set_xlabel(r"$\epsilon$ (pixel /255)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)


def draw_figure(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(3, 2, figsize=(7.15, 6.05), sharey=True)
    metrics = ["cp", "pp"]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    row_labels = {
        "2D": r"2D $28\!\times\!28$" + "\n" + r"$\Delta$ primary metric (pp)",
        "3D": r"3D $28^3$" + "\n" + r"$\Delta$ primary metric (pp)",
        "Tabular": "Tabular\n" + r"$\Delta$ primary metric (pp)",
    }
    panel_index = 0
    for row, modality in enumerate(MODALITY_ORDER):
        for column, metric in enumerate(metrics):
            ax = axes[row, column]
            _plot_panel(ax, summary, modality, metric)
            if row == 0:
                ax.set_title(METRIC_LABELS[metric], pad=8)
            if column == 0:
                ax.set_ylabel(row_labels[modality])
            ax.set_ylim(-18.0, 3.0)
            ax.text(
                -0.12, 1.04, panel_labels[panel_index], transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom", ha="left",
            )
            panel_index += 1

    legend_handles = [
        Line2D(
            [0], [0], color="#A6A6A6", linewidth=0.75, marker="o",
            markersize=2.5, label="Individual dataset",
        ),
        Line2D(
            [0], [0], color=COLORS["cp"], linewidth=2,
            marker="o", label="Across-dataset median CP",
        ),
        Line2D(
            [0], [0], color=COLORS["pp"], linewidth=2,
            linestyle="--", marker="s", label="Across-dataset median PP",
        ),
    ]
    figure.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        frameon=False, bbox_to_anchor=(0.5, 0.012),
    )
    figure.subplots_adjust(
        left=0.13, right=0.98, top=0.94, bottom=0.13,
        wspace=0.18, hspace=0.46,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"perturbation_budget_overview.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-2d", nargs="+", type=Path, default=DEFAULT_2D)
    parser.add_argument("--input-3d", type=Path, default=DEFAULT_3D)
    parser.add_argument("--input-tabular", type=Path, default=DEFAULT_TABULAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    data = load_budget_data(args.input_2d, args.input_3d, args.input_tabular)
    summary = summarize_budget_data(data)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "perturbation_budget_overview_data.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    for path in draw_figure(summary, output_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
