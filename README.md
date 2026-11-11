# DataCook

[![arXiv](https://img.shields.io/badge/arXiv-2403.17755-b31b1b.svg)](https://arxiv.org/abs/2403.17755)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)

Official implementation of **DataCook: Exploring Anti-Adversarial Fingerprints for Healthcare Data Usage Control**, accepted by *Meta-Radiology* (2026).

Sihan Shang, Jun Yan, Danli Shi, Haizhong Zheng, Shuangjia Zheng, Wei Zhao, and Jiancheng Yang.

> The journal DOI has not yet been assigned. The accepted paper and current metadata are available from [arXiv:2403.17755](https://arxiv.org/abs/2403.17755).

## Overview

DataCook explores deployment-stage healthcare data usage control using anti-adversarial fingerprints. A private cooking procedure is applied before data distribution. Models trained on cooked data may preserve predictive utility on similarly cooked inputs while performing less effectively on raw inputs.

The method is a **task-dependent proof of concept**, not a universal mechanism for preventing unauthorized data use. Its behavior depends on the dataset, modality, task, metric, model, and perturbation budget.

![DataCook overview](images/Overview.png)

## Included experiments

| Component | Included scope |
| --- | --- |
| MedMNIST 2D | BloodMNIST, BreastMNIST, RetinaMNIST, PneumoniaMNIST, DermaMNIST, and OrganCMNIST |
| Multi-label imaging | ChestMNIST |
| MedMNIST3D | FractureMNIST3D, NoduleMNIST3D, OrganMNIST3D, and SynapseMNIST3D |
| MedMNIST+ 224x224 | BreastMNIST, PneumoniaMNIST, and RetinaMNIST |
| Structured medical data | Pima Diabetes, Hepatitis C, and Systemic Diseases |
| Evaluation | Accuracy, task-aware AUC, CP, PP, calibration, image quality, runtime, pseudo-label analysis, and paired exploratory statistics |
| Adaptive attacks | JPEG, smoothing, median filtering, augmentation, and limited-data fine-tuning |

The unified configurations in `configs/pipelines/` reproduce DataCook experiments. `attack_type: datacook_adv` selects the adversarial variant. Historical per-dataset launchers remain under `scripts/` for traceability during migration and are not used by the unified quick start below.

## Installation

The experiments were developed with Python 3.8 and PyTorch 2.0. A CUDA-capable GPU is recommended for training and attack evaluation.

```bash
conda create -n datacook python=3.8 -y
conda activate datacook
pip install -r requirements.txt
```

LPIPS is optional and is only required for 2D perceptual image-quality evaluation:

```bash
pip install lpips
```

## Data preparation

### MedMNIST

Install `medmnist` through `requirements.txt`. The pipeline downloads an official split when it is absent from the configured root. Set `dataset.data_path` in each pipeline YAML to the desired MedMNIST directory. The bundled configurations default to:

```text
../datasets/
```

For example, edit `configs/pipelines/bloodmnist.yaml` when using a different location. Both 28x28/28x28x28 and 224x224 variants are obtained through the official [MedMNIST API](https://medmnist.com/).

### Structured medical data

The datasets are not redistributed. After obtaining them from their respective providers, use the following filenames or override `--data-path`:

```text
TableData/dataset/diabetes.csv
TableData/dataset/HepatitisCdata.csv
TableData/dataset/systemic diseases.csv
```

The Systemic Diseases experiment uses restricted-access cohort data; users are responsible for obtaining the required authorization and complying with the provider's terms.

## Quick start

Inspect the commands for the primary 2D suite without running training:

```bash
python run_datacook_suite.py \
  --config configs/pipelines/medmnist_three.yaml \
  --dry-run
```

Run the suite on one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python run_datacook_suite.py \
  --config configs/pipelines/medmnist_three.yaml
```

The pipeline is resumable. Existing complete artifacts are reused. A single dataset, seed, and perturbation budget can be selected for a smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python run_datacook_pipeline.py \
  --config configs/pipelines/bloodmnist.yaml \
  --seeds 0 --epsilons 8
```

Additional suites:

```bash
# Additional 2D datasets
python run_datacook_suite.py --config configs/pipelines/medmnist_extra_three.yaml

# Multi-label ChestMNIST
python run_datacook_pipeline.py --config configs/pipelines/chestmnist.yaml

# Four volumetric datasets
python run_datacook_suite.py --config configs/pipelines/medmnist3d_four.yaml

# Three 224x224 datasets
python run_datacook_suite.py --config configs/pipelines/medmnist_224_three.yaml
```

## Structured-data experiments

The final tabular protocol uses Adam, batch size 32, `epsilon=5`, absolute step size `0.05`, and 300 cooking steps at the representative budget. The same scripts also support the configured perturbation-budget grid.

```bash
python run_tabular_epsilon_ablation.py \
  --config configs/revision/tabular_legacy_params_diabetes.yaml \
  --device cuda

python run_tabular_epsilon_ablation.py \
  --config configs/revision/tabular_legacy_params_hepatitis_c.yaml \
  --device cuda

python run_tabular_epsilon_ablation.py \
  --config configs/revision/tabular_legacy_params_systemic_diseases.yaml \
  --device cuda
```

Use `--data-path /path/to/file.csv` to override the dataset location.

## Evaluation and post-hoc experiments

Each completed imaging run exports three prediction artifacts:

```text
clean.npz                    # clean model evaluated on raw test data
authorized_cooked.npz        # protected model evaluated on cooked test data
unauthorized_raw.npz         # protected model evaluated on raw test data
```

Generate post-hoc configurations from completed runs and execute the enabled analyses:

```bash
python prepare_revision_posthoc.py \
  --suite-config configs/pipelines/medmnist_three.yaml \
  --strict

python run_revision_experiments.py \
  --config configs/revision/generated.yaml \
  --device cuda
```

CP and PP are signed changes relative to the clean reference:

```text
CP = unauthorized_raw - clean
PP = authorized_cooked - clean
```

More negative CP indicates stronger unauthorized degradation. PP closer to zero indicates better preservation of authorized utility. Binary tasks use AUC; multiclass and multi-label tasks use macro-AUC when task-aware AUC results are reported.

Detailed commands, artifact contracts, and experiment notes are provided in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Repository structure

```text
configs/pipelines/       End-to-end dataset and budget configurations
configs/revision/        Evaluation, attack, ablation, and analysis configurations
metrics/                 Classification, CP/PP, and image-quality metrics
attacks/                 Adaptive attack implementations
analysis/                Aggregation and exploratory statistical analysis
revision_experiments/    Shared configuration, I/O, and model utilities
jobs/                    Optional multi-stage job wrappers
tests/                   Regression and smoke tests
```

Generated data, checkpoints, predictions, and results are intentionally excluded from version control.

## Testing

```bash
pip install -r requirements-dev.txt
python -m compileall -q .
pytest -q
```

GPU training is not performed by the unit tests.

## Citation

```bibtex
@misc{shang2026datacook,
  title         = {DataCook: Exploring Anti-Adversarial Fingerprints for Healthcare Data Usage Control},
  author        = {Shang, Sihan and Yan, Jun and Shi, Danli and Zheng, Haizhong and Zheng, Shuangjia and Zhao, Wei and Yang, Jiancheng},
  year          = {2026},
  eprint        = {2403.17755},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2403.17755},
  note          = {Accepted for publication in Meta-Radiology}
}
```

The citation will be updated when the journal DOI and final bibliographic record become available.

## Acknowledgements

Parts of the implementation were adapted from:

- [Unlearnable Examples](https://github.com/HanxunH/Unlearnable-Examples)
- [Unlearnable Clusters](https://github.com/jiamingzhang94/Unlearnable-Clusters)

Please also cite the original MedMNIST datasets and the baseline methods used in an experiment.

## License

This repository is released under the [Apache License 2.0](LICENSE). Dataset access and redistribution remain subject to the licenses and terms of the original data providers.

## Contact

For questions about the code, contact Sihan Shang at [loveligo97@gmail.com](mailto:loveligo97@gmail.com).
