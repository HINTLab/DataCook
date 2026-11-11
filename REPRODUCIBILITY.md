# Reproducing the DataCook experiments

This document describes the public experiment interfaces. YAML files are the source of truth for dataset sizes, seeds, perturbation budgets, models, and output locations.

## 1. Experiment contract

An imaging run contains four stages:

1. Train or restore the surrogate and generate cooked train/test data.
2. Export clean reference predictions.
3. Train or restore the protected victim and export authorized and unauthorized predictions.
4. Compute task-aware metrics and CP/PP.

`run_datacook_pipeline.py` records stage state and reuses complete artifacts. Use `--force` only when an intentional rerun is required.

```bash
python run_datacook_pipeline.py --config CONFIG.yaml --dry-run
python run_datacook_pipeline.py --config CONFIG.yaml
python run_datacook_pipeline.py --config CONFIG.yaml --only metrics
```

Command-line `--seeds` and `--epsilons` values override the YAML grid.

## 2. Imaging suites

| Scope | Configuration |
| --- | --- |
| Primary 2D | `configs/pipelines/medmnist_three.yaml` |
| Additional 2D | `configs/pipelines/medmnist_extra_three.yaml` |
| ChestMNIST multi-label | `configs/pipelines/chestmnist.yaml` |
| MedMNIST3D | `configs/pipelines/medmnist3d_four.yaml` |
| MedMNIST+ 224x224 | `configs/pipelines/medmnist_224_three.yaml` |

Suite files contain relative paths to individual dataset configurations. Run them from the repository root:

```bash
python run_datacook_suite.py --config configs/pipelines/medmnist3d_four.yaml
```

## 3. Prediction artifacts

Prediction files are compressed NPZ files containing:

```text
y_true    Reference labels
y_prob    Softmax or sigmoid probabilities in [0, 1]
```

Accepted shapes are:

- binary: `(N,)`, `(N,1)`, or `(N,2)` probabilities;
- multiclass: `(N,C)` probabilities;
- multi-label: `(N,L)` labels and probabilities.

The three required settings are `clean`, `authorized_cooked`, and `unauthorized_raw`. Existing checkpoints can be exported with:

```bash
python export_revision_predictions.py --config path/to/exports.yaml --device cuda
```

## 4. Unified metrics

```bash
python run_unified_evaluation.py --config configs/revision/metrics.yaml
```

The evaluator supports task accuracy, AUC/macro-AUC, sensitivity, specificity, precision, F1, and ECE where mathematically defined. CP and PP are computed only from metrics whose direction is explicitly supported.

Results are written as long-form CSV files. Missing probabilities, classes, checkpoints, or paired observations produce explicit errors or NaN values; the scripts do not create synthetic result rows.

## 5. Adaptive attacks

```bash
python run_adaptive_attacks.py --config configs/revision/attacks.yaml --device cuda
```

Supported analyses include JPEG compression, Gaussian smoothing, median filtering, strong augmentation, ordinary fine-tuning, and augmentation-based robust fine-tuning. Fine-tuning copies a protected checkpoint and does not overwrite the original model.

## 6. Image quality, pseudo labels, and runtime

```bash
python run_image_quality.py --config configs/revision/image_quality.yaml
python run_pseudo_label_analysis.py --config configs/revision/pseudo.yaml
python run_runtime_analysis.py --config configs/revision/runtime.yaml --device cuda
```

LPIPS is optional and applies only to compatible 2D images. Volumetric quality reports SSIM, PSNR, L2, and Linf without LPIPS. Runtime configurations record the device, warm-up, repetitions, sample count, batch size, and aggregation rule.

## 7. Perturbation-budget ablations

Imaging ablations consume completed prediction artifacts:

```bash
python run_epsilon_ablation.py --config configs/revision/epsilon.yaml
```

Structured-data ablations train and evaluate directly from the configured CSV:

```bash
python run_tabular_epsilon_ablation.py \
  --config configs/revision/tabular_legacy_params_diabetes.yaml \
  --device cuda
```

At the representative tabular budget, the final configuration is `epsilon=5`, absolute step size `0.05`, and 300 steps. The data are standardized features, and only the explicitly configured feature indices are cookable.

## 8. Statistical analysis

```bash
python -m analysis.aggregate_results --config configs/revision/statistics.yaml
python -m analysis.statistical_tests --config configs/revision/statistics.yaml
```

Paired tests match observations by experiment identifiers such as dataset, model, seed, and perturbation budget. Holm correction is applied within the configured families. These tests are exploratory; conclusions should be reported together with means, standard deviations, confidence intervals, effect sizes, and the number of paired observations.

## 9. Generated post-hoc configuration

After completing a suite, generate configs that point to its actual checkpoints and artifacts:

```bash
python prepare_revision_posthoc.py \
  --suite-config configs/pipelines/medmnist_three.yaml \
  --strict
```

Then run the enabled tasks:

```bash
python run_revision_experiments.py \
  --config configs/revision/generated.yaml \
  --device cuda
```

`--strict` is recommended for final analyses because it rejects incomplete experiment sets.

## 10. Outputs excluded from Git

The following directories are generated locally and must not be committed:

```text
data/
datasets/
experiments/
artifacts/
results/
outputs/
revision_logs/
```

Do not commit patient-level data, UK Biobank-derived tables, checkpoints, prediction NPZ files, or server-specific synchronization scripts.
