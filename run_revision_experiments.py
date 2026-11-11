#!/usr/bin/env python3
"""Master entry point for all DataCook revision experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from revision_experiments.common import load_config, resolve_path, run_prepare_commands, setup_logging


TASK_COMMANDS = {
    "metrics": [["run_unified_evaluation.py"]],
    "attacks": [["run_adaptive_attacks.py"]],
    "finetune_attacks": [["run_adaptive_attacks.py"]],
    "finetune_budget_attacks": [["run_adaptive_attacks.py"]],
    "epsilon": [["run_epsilon_ablation.py"]],
    "pseudo": [["run_pseudo_label_analysis.py"]],
    "statistics": [["-m", "analysis.aggregate_results"], ["-m", "analysis.statistical_tests"]],
    "runtime": [["run_runtime_analysis.py"]],
    "image_quality": [["run_image_quality.py"]],
    "final_analysis": [["-m", "analysis.finalize_revision_results"]],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/revision.yaml")
    parser.add_argument("--tasks", nargs="*", choices=TASK_COMMANDS, help="Run only these tasks; enabled flags are ignored.")
    parser.add_argument("--prepare", action="store_true", help="Run each task config's explicit prepare_commands first.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("revision-master", args.verbose)
    config, base_dir = load_config(args.config)
    project_root = Path(__file__).resolve().parent
    requested = set(args.tasks or [])
    failures = []
    for task_name, commands in TASK_COMMANDS.items():
        task_config = config.get("tasks", {}).get(task_name, {})
        if args.tasks is None and not task_config.get("enabled", False):
            logger.info("Skipping disabled task %s", task_name)
            continue
        if requested and task_name not in requested:
            continue
        if "config" not in task_config:
            raise ValueError(f"Task {task_name!r} has no config path in the master config.")
        task_config_path = resolve_path(task_config["config"], base_dir)
        try:
            if args.prepare:
                child_config, child_base = load_config(task_config_path)
                run_prepare_commands(child_config, child_base, logger)
            for command in commands:
                if command[0] == "-m":
                    full = [sys.executable, *command, "--config", str(task_config_path)]
                else:
                    full = [sys.executable, str(project_root / command[0]), "--config", str(task_config_path)]
                if args.verbose:
                    full.append("--verbose")
                if args.device and command[0] in {"run_adaptive_attacks.py", "run_runtime_analysis.py"}:
                    full.extend(["--device", args.device])
                logger.info("Running task %s: %s", task_name, " ".join(full))
                subprocess.run(full, cwd=project_root, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append((task_name, exc.returncode))
            logger.error("Task %s failed with exit code %s", task_name, exc.returncode)
            if not args.continue_on_error:
                raise
    if failures:
        raise SystemExit("Revision tasks failed: " + ", ".join(f"{name}({code})" for name, code in failures))
    logger.info("All selected revision tasks completed.")


if __name__ == "__main__":
    main()
