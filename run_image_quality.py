#!/usr/bin/env python3
"""Task 8: compute image quality and imperceptibility metrics."""

from __future__ import annotations

import argparse

from metrics import compute_image_quality_metrics
from revision_experiments.common import load_array, load_config, metadata, require_records, setup_logging, write_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("image-quality", args.verbose)
    config, base_dir = load_config(args.config)
    rows = []
    for record in require_records(config):
        images = record.get("images", {})
        if "raw" not in images or "cooked" not in images:
            raise ValueError("Each image-quality experiment requires images.raw and images.cooked.")
        raw = load_array(images["raw"], base_dir)
        cooked = load_array(images["cooked"], base_dir)
        logger.info("Computing image quality for %s (%d samples)", record["metadata"]["dataset"], len(raw))
        values = compute_image_quality_metrics(
            raw, cooked, data_range=record.get("data_range"),
            compute_lpips=bool(record.get("compute_lpips", True)), device=record.get("device"),
        )
        rows.append({**metadata(record, setting="image_quality"), **values})
    output = config.get("output", "results/image_quality/quality_metrics.csv")
    logger.info("Wrote %s", write_rows(rows, output))


if __name__ == "__main__":
    main()
