# Historical launchers

This directory contains the original per-dataset shell launchers for DataCook,
DataCook (Adv), and the comparison methods. They are retained temporarily for
historical traceability and baseline migration.

For current DataCook experiments, use the validated YAML-driven entry points
documented in the repository root:

```bash
python run_datacook_pipeline.py --config configs/pipelines/bloodmnist.yaml
python run_datacook_suite.py --config configs/pipelines/medmnist_three.yaml
```

The historical scripts are not the source of truth for the accepted-paper
experiments. Do not mix their arguments or output layouts with the unified
pipeline. They can be removed after every comparison-method launcher needed for
reproduction has been migrated and validated against its historical output.
