#!/usr/bin/env python3
"""Resumable one-click DataCook main experiment and revision evaluation pipeline."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
STAGES = ("perturbation", "clean_victim", "protected", "metrics")


def _path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", str(value)).strip("-")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _complete(paths: list[Path]) -> bool:
    return all(path.is_file() if path.suffix else path.is_dir() for path in paths)


def _run(command: list[str], log_path: Path, logger: logging.Logger, dry_run: bool) -> None:
    rendered = " ".join(command)
    logger.info("Command: %s", rendered)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{dt.datetime.now().isoformat()}] {rendered}\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _main_command(
    python: str,
    *,
    model: str,
    config_path: Path,
    exp_name: Path,
    dataset_name: str,
    train_type: str,
    train_path: Path,
    test_type: str,
    test_path: Path,
    recover_type: str,
    recover_path: Path,
    prediction_dir: Path,
    role: str,
    seed: int,
    train_batch_size: int,
    eval_batch_size: int,
    workers: int,
    checkpoint: Path,
    high_resolution: bool = False,
) -> list[str]:
    command = [
        python, "-u", "main.py",
        "--version", model,
        "--exp_name", str(exp_name),
        "--config_path", str(config_path),
        "--train_data_type", train_type,
        "--train_data_path", str(train_path),
        "--test_data_type", test_type,
        "--test_data_path", str(test_path),
        "--recover_test_data_type", recover_type,
        "--recover_test_data_path", str(recover_path),
        "--train_batch_size", str(train_batch_size),
        "--eval_batch_size", str(eval_batch_size),
        "--num_of_workers", str(workers),
        "--seed", str(seed),
        "--deterministic",
        "--train",
        "--export_predictions_dir", str(prediction_dir),
        "--evaluation_role", role,
    ]
    if checkpoint.is_file():
        command.append("--load_model")
    if high_resolution:
        command.append("--high_resolution")
    return command


def _run_one(config: dict[str, Any], seed: int, epsilon: float, args, logger) -> dict[str, Path]:
    dataset_cfg = config["dataset"]
    experiment_cfg = config.get("experiment", {})
    model_cfg = config["models"]
    runtime_cfg = config.get("runtime", {})
    dataset_name = dataset_cfg["name"]
    high_resolution = bool(dataset_cfg.get("high_resolution", False))
    poison_name = dataset_cfg.get("poison_name", "Poison" + dataset_name)
    surrogate = model_cfg.get("surrogate", "resnet18")
    victim = model_cfg.get("victim", surrogate)
    epsilon_scale = experiment_cfg.get("epsilon_scale", "pixel")
    normalized_epsilon = epsilon / 255.0 if epsilon_scale == "pixel" else epsilon
    step_size = experiment_cfg.get("step_size")
    if step_size is None:
        step_size = epsilon * float(experiment_cfg.get("step_size_ratio", 0.2))

    epsilon_label = f"{epsilon:g}"
    run_name = _slug(
        f"{dataset_name}-{experiment_cfg.get('attack_type', 'datacook')}-"
        f"sur={surrogate}-vic={victim}-seed={seed}-eps={epsilon_label}"
    )
    experiment_root = _path(config.get("experiment_root", "experiments/revision_pipeline"))
    result_root = _path(config.get("result_root", "results/revision_pipeline"))
    run_dir = experiment_root / run_name
    prediction_dir = run_dir / "predictions"
    log_dir = run_dir / "pipeline_logs"
    state_path = run_dir / "pipeline_state.json"
    is_3d = str(dataset_cfg.get("modality", "")).lower() in {
        "3d", "3d_image", "3d_volume", "volume"
    } or len(dataset_cfg.get("noise_shape", [])) == 5
    task_type = str(dataset_cfg.get("task_type", "multiclass"))
    normalized_task = task_type.replace("-", "").replace("_", "").lower()
    is_multilabel = normalized_task == "multilabel"
    cooked_as_npz = is_3d or is_multilabel
    cooked_train_dir = run_dir / surrogate / "ae"
    cooked_test_dir = run_dir / surrogate / "ae_test"
    cooked_train = cooked_train_dir / "perturb_data.npz" if cooked_as_npz else cooked_train_dir
    cooked_test = cooked_test_dir / "perturb_data.npz" if cooked_as_npz else cooked_test_dir
    python = str(runtime_cfg.get("python", sys.executable))
    config_path = _path(dataset_cfg["config_path"])
    for model_name in {surrogate, victim}:
        model_config = config_path / f"{model_name}.yaml"
        if not model_config.is_file():
            raise FileNotFoundError(f"Model config does not exist: {model_config}")
    raw_data_path = _path(dataset_cfg.get("data_path", "../datasets"))
    noise_shape = dataset_cfg.get("noise_shape")
    test_noise_shape = dataset_cfg.get("test_noise_shape")
    if not noise_shape or not test_noise_shape:
        raise ValueError("dataset.noise_shape and dataset.test_noise_shape are required for the current generator.")
    expected_size = 224 if high_resolution else None
    if expected_size is not None and (
        len(noise_shape) != 4
        or len(test_noise_shape) != 4
        or list(noise_shape[-2:]) != [expected_size, expected_size]
        or list(test_noise_shape[-2:]) != [expected_size, expected_size]
    ):
        raise ValueError(
            "High-resolution pipelines require 4D noise_shape/test_noise_shape "
            "ending in [224, 224]."
        )
    num_classes = int(dataset_cfg["num_classes"])
    train_batch_size = int(runtime_cfg.get("train_batch_size", 128))
    eval_batch_size = int(runtime_cfg.get("eval_batch_size", train_batch_size))
    workers = int(runtime_cfg.get("num_workers", 8))

    state = {
        "run_name": run_name,
        "dataset": dataset_name,
        "seed": seed,
        "epsilon": epsilon,
        "epsilon_normalized": normalized_epsilon,
        "updated_at": dt.datetime.now().isoformat(),
        "stages": {},
    }
    if state_path.is_file():
        state.update(json.loads(state_path.read_text(encoding="utf-8")))
    logger.info("===== Run %s =====", run_name)

    selected = set(args.only or STAGES)

    def stage(name: str, command: list[str], expected: list[Path]) -> None:
        if name not in selected:
            logger.info("Skipping unselected stage %s", name)
            return
        if not args.force and _complete(expected):
            logger.info("Stage %s already complete; reusing artifacts.", name)
            state["stages"][name] = {"status": "reused", "updated_at": dt.datetime.now().isoformat()}
            if not args.dry_run:
                _write_json(state_path, state)
            return
        logger.info("Starting stage %s", name)
        state["stages"][name] = {"status": "running", "updated_at": dt.datetime.now().isoformat()}
        if not args.dry_run:
            _write_json(state_path, state)
        try:
            _run(command, log_dir / f"{name}.log", logger, args.dry_run)
            if not args.dry_run and not _complete(expected):
                missing = [str(path) for path in expected if not (path.is_file() if path.suffix else path.is_dir())]
                raise RuntimeError(f"Stage {name} exited successfully but outputs are missing: {missing}")
            state["stages"][name] = {"status": "complete", "updated_at": dt.datetime.now().isoformat()}
        except Exception as exc:
            state["stages"][name] = {"status": "failed", "error": str(exc), "updated_at": dt.datetime.now().isoformat()}
            if not args.dry_run:
                _write_json(state_path, state)
            raise
        if not args.dry_run:
            _write_json(state_path, state)

    perturb_command = [
        python, "-u", "perturbation.py",
        "--seed", str(seed),
        "--version", surrogate,
        "--exp_name", str(run_dir),
        "--config_path", str(config_path),
        "--train_data_type", dataset_name,
        "--train_data_path", str(raw_data_path),
        "--test_data_type", dataset_name,
        "--test_data_path", str(raw_data_path),
        "--recover_test_data_type", dataset_name,
        "--recover_test_data_path", str(raw_data_path),
        "--noise_shape", *map(str, noise_shape),
        "--test_noise_shape", *map(str, test_noise_shape),
        "--epsilon", str(epsilon),
        "--epsilon_scale", epsilon_scale,
        "--step_size", str(step_size),
        "--num_steps", str(experiment_cfg.get("num_steps", 20)),
        "--attack_type", experiment_cfg.get("attack_type", "datacook"),
        "--universal_stop_error", str(experiment_cfg.get("universal_stop_error", 0.01)),
        "--train_batch_size", str(train_batch_size),
        "--eval_batch_size", str(eval_batch_size),
        "--num_of_workers", str(workers),
        "--deterministic",
        "--export_predictions_dir", str(prediction_dir),
    ]
    if high_resolution:
        perturb_command.append("--high_resolution")
    stage(
        "perturbation",
        perturb_command,
        [run_dir / "perturbation.pt", run_dir / "test_perturbation.pt", cooked_train, cooked_test, prediction_dir / "clean.npz"],
    )

    if victim != surrogate:
        clean_exp = run_dir / "clean_victim"
        clean_checkpoint = clean_exp / victim / "checkpoints" / f"{victim}.pth"
        clean_command = _main_command(
            python, model=victim, config_path=config_path, exp_name=clean_exp,
            dataset_name=dataset_name, train_type=dataset_name, train_path=raw_data_path,
            test_type=dataset_name, test_path=raw_data_path, recover_type=dataset_name,
            recover_path=raw_data_path, prediction_dir=prediction_dir, role="clean",
            seed=seed, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
            workers=workers, checkpoint=clean_checkpoint, high_resolution=high_resolution,
        )
        stage("clean_victim", clean_command, [clean_checkpoint, prediction_dir / "clean.npz"])
    else:
        logger.info("Victim equals surrogate; clean predictions are reused from perturbation stage.")

    protected_exp = run_dir / "protected"
    protected_checkpoint = protected_exp / victim / "checkpoints" / f"{victim}.pth"
    protected_command = _main_command(
        python, model=victim, config_path=config_path, exp_name=protected_exp,
        dataset_name=dataset_name, train_type=poison_name, train_path=cooked_train,
        test_type=dataset_name, test_path=raw_data_path, recover_type=poison_name,
        recover_path=cooked_test, prediction_dir=prediction_dir, role="protected",
        seed=seed, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
        workers=workers, checkpoint=protected_checkpoint, high_resolution=high_resolution,
    )
    stage(
        "protected",
        protected_command,
        [protected_checkpoint, prediction_dir / "authorized_cooked.npz", prediction_dir / "unauthorized_raw.npz"],
    )

    metrics_dir = result_root / run_name / "metrics"
    metrics_config = run_dir / "generated_metrics.yaml"
    metrics_payload = {
        "experiments": [{
            "metadata": {
                "dataset": dataset_name,
                "modality": dataset_cfg.get(
                    "modality", "high_resolution" if high_resolution else ("3d_volume" if is_3d else "2d_image")
                ),
                "task_type": task_type,
                "method": experiment_cfg.get("attack_type", "datacook"),
                "surrogate_model": surrogate,
                "victim_model": victim,
                "seed": seed,
                "epsilon": normalized_epsilon,
            },
            "num_classes": num_classes,
            "threshold": experiment_cfg.get("threshold", 0.5),
            "metric_names": dataset_cfg.get("metric_names"),
            "predictions": {
                "clean": str(prediction_dir / "clean.npz"),
                "authorized_cooked": str(prediction_dir / "authorized_cooked.npz"),
                "unauthorized_raw": str(prediction_dir / "unauthorized_raw.npz"),
            },
        }],
        "outputs": {
            "main_metrics": str(metrics_dir / "main_metrics.csv"),
            "cp_pp": str(metrics_dir / "cp_pp_summary.csv"),
        },
    }
    if not args.dry_run:
        metrics_config.parent.mkdir(parents=True, exist_ok=True)
        metrics_config.write_text(yaml.safe_dump(metrics_payload, sort_keys=False), encoding="utf-8")
    metrics_command = [python, "run_unified_evaluation.py", "--config", str(metrics_config)]
    stage("metrics", metrics_command, [metrics_dir / "main_metrics.csv", metrics_dir / "cp_pp_summary.csv"])
    logger.info("Completed %s; results: %s", run_name, metrics_dir)
    return {
        "main_metrics": metrics_dir / "main_metrics.csv",
        "cp_pp": metrics_dir / "cp_pp_summary.csv",
        "result_root": result_root,
    }


def _concatenate_csv(inputs: list[Path], output: Path) -> None:
    existing = [path for path in inputs if path.is_file()]
    if not existing:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = None
    with output.open("w", encoding="utf-8", newline="") as output_handle:
        writer = None
        for path in existing:
            with path.open("r", encoding="utf-8", newline="") as input_handle:
                reader = csv.DictReader(input_handle)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
                    writer.writeheader()
                elif reader.fieldnames != fieldnames:
                    raise ValueError(f"Cannot concatenate CSV files with different schemas: {path}")
                assert writer is not None
                writer.writerows(reader)


def _discover_run_results(result_root: Path, filename: str) -> list[Path]:
    """Find per-run results so summaries include every dataset/config run.

    A pipeline invocation used to summarize only the seed/epsilon combinations
    from the current config.  That caused a later dataset invocation to replace
    the shared summary with results from that dataset alone.  Scanning the
    per-run directories makes sequential multi-dataset execution resumable and
    keeps one cumulative summary.
    """
    return sorted(
        path for path in result_root.glob(f"*/metrics/{filename}")
        if path.is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running or writing state.")
    parser.add_argument("--force", action="store_true", help="Rerun selected stages even if outputs exist.")
    parser.add_argument("--only", nargs="+", choices=STAGES, help="Run only selected stages.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override the seed list in the YAML config.")
    parser.add_argument("--epsilons", nargs="+", type=float, help="Override the epsilon list in the YAML config.")
    args = parser.parse_args()
    config_path = _path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Pipeline config must be a YAML mapping.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("datacook-pipeline")
    if not args.dry_run:
        python = str(config.get("runtime", {}).get("python", sys.executable))
        preflight = subprocess.run(
            [python, "-c", (
                "import torch, torchvision, mlconfig, medmnist, sklearn, skimage, pandas, yaml, timm; "
                "print('PyTorch:', torch.__version__); "
                "print('timm:', timm.__version__); "
                "print('CUDA available:', torch.cuda.is_available()); "
                "print('Device count:', torch.cuda.device_count())"
            )],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        if preflight.stdout:
            logger.info("Environment preflight:\n%s", preflight.stdout.rstrip())
        if preflight.returncode:
            raise RuntimeError(
                "Environment preflight failed. Install requirements in the configured Python environment.\n"
                + preflight.stderr
            )
    seeds = args.seeds if args.seeds is not None else config.get("seeds", [0])
    epsilons = (
        args.epsilons if args.epsilons is not None
        else config.get("epsilons", [config.get("experiment", {}).get("epsilon", 8)])
    )
    completed_runs = []
    for seed in seeds:
        for epsilon in epsilons:
            completed_runs.append(_run_one(config, int(seed), float(epsilon), args, logger))

    should_summarize = not args.dry_run and (args.only is None or "metrics" in args.only)
    if should_summarize and completed_runs:
        result_root = completed_runs[0]["result_root"]
        summary_dir = result_root / "summary"
        combined_main = summary_dir / "main_metrics.csv"
        combined_cp = summary_dir / "cp_pp_summary.csv"
        _concatenate_csv(_discover_run_results(result_root, "main_metrics.csv"), combined_main)
        _concatenate_csv(_discover_run_results(result_root, "cp_pp_summary.csv"), combined_cp)
        if combined_cp.is_file():
            statistics_config = summary_dir / "statistics.yaml"
            aggregated_output = summary_dir / "aggregated_results.csv"
            statistics_config.write_text(yaml.safe_dump({
                "inputs": [{
                    "path": str(combined_cp),
                    "experiment_type": "main",
                    "value_columns": ["cp", "pp"],
                }],
                "aggregated_output": str(aggregated_output),
            }, sort_keys=False), encoding="utf-8")
            python = str(config.get("runtime", {}).get("python", sys.executable))
            _run(
                [python, "-m", "analysis.aggregate_results", "--config", str(statistics_config)],
                summary_dir / "aggregate.log",
                logger,
                False,
            )
        logger.info("Combined results written to %s", summary_dir)


if __name__ == "__main__":
    main()
