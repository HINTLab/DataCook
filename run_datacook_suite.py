#!/usr/bin/env python3
"""Run several resumable DataCook pipeline configurations sequentially."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
STAGES = ("perturbation", "clean_victim", "protected", "metrics")


def _resolve(path: str | Path, base: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def load_pipeline_configs(suite_path: Path) -> list[Path]:
    payload = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Suite config must be a YAML mapping.")
    entries = payload.get("pipelines")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Suite config requires a non-empty `pipelines` list.")
    configs = [_resolve(entry, suite_path.parent) for entry in entries]
    missing = [str(path) for path in configs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Pipeline config(s) do not exist: " + ", ".join(missing))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Suite YAML containing a `pipelines` list.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="+", choices=STAGES)
    parser.add_argument("--seeds", nargs="+", type=int, help="Override seeds in every pipeline config.")
    parser.add_argument("--epsilons", nargs="+", type=float, help="Override epsilons in every pipeline config.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("datacook-suite")
    suite_path = _resolve(args.config, PROJECT_ROOT)
    configs = load_pipeline_configs(suite_path)
    logger.info("Suite contains %d dataset pipelines.", len(configs))

    for index, pipeline_config in enumerate(configs, start=1):
        command = [
            sys.executable,
            "-u",
            "run_datacook_pipeline.py",
            "--config",
            str(pipeline_config),
        ]
        if args.dry_run:
            command.append("--dry-run")
        if args.force:
            command.append("--force")
        if args.only:
            command.extend(["--only", *args.only])
        if args.seeds is not None:
            command.extend(["--seeds", *map(str, args.seeds)])
        if args.epsilons is not None:
            command.extend(["--epsilons", *map(str, args.epsilons)])
        logger.info("===== Dataset pipeline %d/%d: %s =====", index, len(configs), pipeline_config.name)
        logger.info("Command: %s", " ".join(command))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    logger.info("All %d dataset pipelines completed successfully.", len(configs))


if __name__ == "__main__":
    main()
