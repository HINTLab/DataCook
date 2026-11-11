#!/usr/bin/env python3
"""Export standardized prediction NPZ files from existing PyTorch checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from revision_experiments.common import load_config, require_records, resolve_path, setup_logging
from revision_experiments.modeling import import_object, load_model, load_xy_artifact, predict_probabilities, probabilities_from_logits


def _from_loader(model, data_spec, task_type, device):
    factory = import_object(data_spec["loader_factory"])
    loader = factory(**dict(data_spec.get("kwargs", {})))
    labels, probabilities = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) < 2:
                raise ValueError("Loader batches must contain at least inputs and labels.")
            logits = model(batch[0].to(device))
            probabilities.append(probabilities_from_logits(logits, task_type).cpu())
            labels.append(batch[1].detach().cpu())
    return torch.cat(labels).numpy(), torch.cat(probabilities).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("export-predictions", args.verbose)
    config, base_dir = load_config(args.config)
    for record in require_records(config, key="exports"):
        task_type = record["metadata"]["task_type"]
        model = load_model(record["model"], base_dir, args.device)
        data_spec = record["data"]
        if "loader_factory" in data_spec:
            labels, probability = _from_loader(model, data_spec, task_type, args.device)
        else:
            inputs, labels = load_xy_artifact(data_spec, base_dir)
            probability = predict_probabilities(
                model, inputs, task_type=task_type, device=args.device,
                batch_size=int(record.get("batch_size", 128)),
            )
        output = resolve_path(record["output"], base_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, y_true=labels, y_prob=probability)
        logger.info("Wrote %s (%d samples)", output, len(labels))


if __name__ == "__main__":
    main()
