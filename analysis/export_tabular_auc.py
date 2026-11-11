#!/usr/bin/env python3
"""Export paper-ready AUC summaries for the tabular DataCook experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    "downloaded_tabular_legacy_extended/results/"
    "revision_pipeline_tabular_legacy_extended/summary/cp_pp_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "downloaded_tabular_legacy_extended/results/"
    "revision_pipeline_tabular_legacy_extended/paper_tables"
)

# AUC must be selected according to task type. Binary tasks expose ``auc``;
# multiclass and multilabel tasks expose macro-averaged one-vs-rest AUC.
AUC_METRIC_BY_TASK = {
    "binary": "auc",
    "multiclass": "macro_auc",
    "multilabel": "macro_auc",
}

VALUE_COLUMNS = ["clean", "authorized_cooked", "unauthorized_raw", "cp", "pp"]


def select_auc_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return task-appropriate AUC rows and fail on missing task coverage."""
    required = {"dataset", "task_type", "seed", "epsilon", "metric", *VALUE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"AUC input is missing columns: {sorted(missing)}")

    selected_frames = []
    for task_type, metric in AUC_METRIC_BY_TASK.items():
        selected_frames.append(
            frame[frame["task_type"].eq(task_type) & frame["metric"].eq(metric)].copy()
        )
    selected = pd.concat(selected_frames, ignore_index=True)
    if selected.empty:
        raise ValueError("No task-appropriate AUC rows were found.")

    expected_datasets = set(frame["dataset"].dropna().astype(str))
    selected_datasets = set(selected["dataset"].dropna().astype(str))
    missing_datasets = expected_datasets.difference(selected_datasets)
    if missing_datasets:
        raise ValueError(
            "No task-appropriate AUC rows for datasets: "
            f"{sorted(missing_datasets)}"
        )
    if selected[VALUE_COLUMNS].isna().any().any():
        counts = selected[VALUE_COLUMNS].isna().sum()
        raise ValueError(f"Selected AUC rows contain missing values:\n{counts}")
    return selected


def summarize_auc(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate AUC, CP and PP over seeds for each dataset and budget."""
    grouped = frame.groupby(
        ["dataset", "task_type", "metric", "epsilon"], dropna=False
    )[VALUE_COLUMNS].agg(["mean", "std", "count"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    result = grouped.reset_index().rename(columns={"metric": "auc_metric"})
    count_columns = [f"{column}_count" for column in VALUE_COLUMNS]
    if not (result[count_columns].nunique(axis=1) == 1).all():
        raise ValueError("AUC value columns have inconsistent seed counts.")
    result.insert(4, "num_seeds", result[count_columns[0]].astype(int))
    return result.drop(columns=count_columns)


def _latex_number(mean: float, std: float, *, difference: bool = False) -> str:
    scale = 100.0
    return f"${mean * scale:.2f}\\pm{std * scale:.2f}$"


def render_latex_table(dataset: str, frame: pd.DataFrame) -> str:
    """Render one self-contained LaTeX table for a single dataset."""
    selected = frame[frame["dataset"].eq(dataset)].sort_values("epsilon")
    if selected.empty:
        raise ValueError(f"No AUC summary rows for {dataset}.")
    metric = str(selected["auc_metric"].iloc[0]).replace("_", "\\_")
    rows = []
    for row in selected.itertuples(index=False):
        cells = [f"{row.epsilon:g}"]
        for value in VALUE_COLUMNS:
            cells.append(
                _latex_number(
                    getattr(row, f"{value}_mean"),
                    getattr(row, f"{value}_std"),
                    difference=value in {"cp", "pp"},
                )
            )
        rows.append(" & ".join(cells) + r" \\")
    body = "\n".join(rows)
    label = dataset.lower().replace("_", "-")
    return rf"""\begin{{table*}}[t]
\small
\caption{{\textbf{{Tabular AUC Results on {dataset}.}} Values are mean$\pm$standard deviation over eight seeds. The task-appropriate AUC metric is \texttt{{{metric}}}; CP and PP are reported as percentage-point changes.}}
\label{{tab:tabular-auc-{label}}}
\centering
\begin{{tabular}}{{rrrrrr}}
\toprule
$\epsilon$ & Clean & Authorized & Unauthorized & CP$\downarrow$ & PP$\uparrow$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    frame = pd.read_csv(source)
    selected = select_auc_rows(frame)
    summary = summarize_auc(selected)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "table_tabular_auc_all.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    for dataset in summary["dataset"].drop_duplicates():
        path = output_dir / f"table_tabular_auc_{dataset.lower()}.tex"
        path.write_text(render_latex_table(dataset, summary), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
