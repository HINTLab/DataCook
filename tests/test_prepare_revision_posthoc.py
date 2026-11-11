import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from prepare_revision_posthoc import (
    RunSpec,
    _cooked_index,
    _load_official_split,
    export_cooked_test,
    generate_configs,
    load_run_specs,
)


def test_suite_expands_to_45_expected_runs():
    specs = load_run_specs(Path("configs/pipelines/medmnist_three.yaml").resolve())
    assert len(specs) == 45
    assert {spec.dataset for spec in specs} == {"BloodMNIST", "BreastMNIST", "RetinaMNIST"}
    assert {spec.seed for spec in specs} == {0, 1, 2}
    assert {spec.epsilon_pixel for spec in specs} == {2.0, 4.0, 8.0, 12.0, 16.0}


def test_suite_cli_seed_override_discovers_runs_outside_base_yaml():
    specs = load_run_specs(
        Path("configs/pipelines/medmnist_three.yaml").resolve(),
        selected_seeds={3, 4, 5, 6, 7},
    )
    assert len(specs) == 75
    assert {spec.seed for spec in specs} == {3, 4, 5, 6, 7}
    assert {spec.epsilon_pixel for spec in specs} == {2.0, 4.0, 8.0, 12.0, 16.0}


def test_suite_cli_seed_and_epsilon_overrides_match_attack_extension():
    specs = load_run_specs(
        Path("configs/pipelines/medmnist_three.yaml").resolve(),
        selected_datasets={"RetinaMNIST"},
        selected_seeds=set(range(8)),
        selected_epsilons={12.0},
    )
    assert len(specs) == 8
    assert {spec.seed for spec in specs} == set(range(8))
    assert {spec.epsilon_pixel for spec in specs} == {12.0}


def test_extra_2d_suite_expands_to_120_expected_runs():
    specs = load_run_specs(Path("configs/pipelines/medmnist_extra_three.yaml").resolve())
    assert len(specs) == 120
    assert {spec.dataset for spec in specs} == {
        "PneumoniaMNIST", "DermaMNIST", "OrganCMNIST",
    }
    assert {spec.seed for spec in specs} == set(range(8))
    assert {spec.epsilon_pixel for spec in specs} == {2.0, 4.0, 8.0, 12.0, 16.0}
    counts = {spec.dataset: (spec.test_count, spec.num_classes) for spec in specs}
    assert counts == {
        "PneumoniaMNIST": (624, 2),
        "DermaMNIST": (2005, 7),
        "OrganCMNIST": (8216, 11),
    }


def test_3d_suite_expands_to_36_expected_runs():
    specs = load_run_specs(Path("configs/pipelines/medmnist3d_four.yaml").resolve())
    assert len(specs) == 36
    assert {spec.dataset for spec in specs} == {
        "OrganMNIST3D", "NoduleMNIST3D",
        "FractureMNIST3D", "SynapseMNIST3D",
    }
    assert {spec.modality for spec in specs} == {"3d_volume"}
    assert {spec.seed for spec in specs} == {0, 1, 2}
    assert {spec.epsilon_pixel for spec in specs} == {8.0, 16.0, 32.0}


def test_official_3d_split_uses_volume_transform(monkeypatch, tmp_path):
    constructor_kwargs = {}

    class FakeVolumeDataset:
        def __init__(self, **kwargs):
            constructor_kwargs.update(kwargs)
            self.transform = kwargs["transform"]

        def __iter__(self):
            for label in (0, 1):
                volume = np.full((1, 4, 4, 4), label, dtype=np.uint8)
                yield self.transform(volume), np.array([label])

    fake_medmnist = types.ModuleType("medmnist")
    fake_medmnist.INFO = {
        "toymnist3d": {"python_class": "FakeVolumeDataset"},
    }
    fake_medmnist.FakeVolumeDataset = FakeVolumeDataset
    monkeypatch.setitem(sys.modules, "medmnist", fake_medmnist)
    spec = RunSpec(
        dataset="ToyMNIST3D", modality="3d_volume", task_type="binary",
        num_classes=2, surrogate="resnet18_3d", victim="resnet18_3d",
        method="datacook", seed=0, epsilon_pixel=8.0, epsilon=8 / 255,
        run_name="ToyMNIST3D-run", run_dir=tmp_path, model_config=tmp_path / "model.yaml",
        data_path=tmp_path, test_count=2,
    )

    images, labels = _load_official_split(spec, "test")

    assert images.shape == (2, 1, 4, 4, 4)
    assert images.dtype == np.float32
    assert labels.tolist() == [0, 1]
    assert constructor_kwargs["as_rgb"] is False
    assert constructor_kwargs["transform"].__class__.__name__ == "Transform3D"


def test_cooked_export_restores_global_order_and_checks_labels(tmp_path):
    raw = np.zeros((3, 3, 2, 2), dtype=np.float32)
    labels = np.array([1, 0, 1], dtype=np.int64)
    cooked_dir = tmp_path / "ae_test"
    values = {0: 32, 1: 96, 2: 224}
    for index in (2, 0, 1):
        folder = cooked_dir / str(labels[index])
        folder.mkdir(parents=True, exist_ok=True)
        image = np.full((2, 2, 3), values[index], dtype=np.uint8)
        Image.fromarray(image).save(folder / f"{index % 2}_{index}.png")

    output = tmp_path / "cooked.npz"
    export_cooked_test(cooked_dir, raw, labels, output, force=False, logger=_NullLogger())
    with np.load(output, allow_pickle=False) as artifact:
        assert artifact["labels"].tolist() == [1, 0, 1]
        assert np.allclose(artifact["images"][:, 0, 0, 0], np.array([32, 96, 224]) / 255)


def test_cooked_export_recovers_from_saved_test_perturbation(tmp_path):
    raw = np.full((2, 3, 2, 2), 0.5, dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    perturbation = torch.from_numpy(np.stack([
        np.full((3, 2, 2), 0.01, dtype=np.float32),
        np.full((3, 2, 2), -0.02, dtype=np.float32),
    ]))
    perturbation_path = tmp_path / "test_perturbation.pt"
    torch.save(perturbation, perturbation_path)
    cooked_dir = tmp_path / "empty_ae_test"
    cooked_dir.mkdir()
    output = tmp_path / "recovered.npz"

    export_cooked_test(
        cooked_dir,
        raw,
        labels,
        output,
        force=False,
        logger=_NullLogger(),
        perturbation_path=perturbation_path,
    )

    expected = np.floor(np.clip(raw + perturbation.numpy(), 0, 1) * 255 + 0.5) / 255
    with np.load(output, allow_pickle=False) as artifact:
        np.testing.assert_allclose(artifact["images"], expected, atol=1e-7)
        np.testing.assert_array_equal(artifact["labels"], labels)


def test_generated_configs_include_only_task_ready_runs(tmp_path):
    run_dir = tmp_path / "experiments" / "Toy-run"
    config_dir = tmp_path / "configs" / "revision" / "generated"
    artifact_root = tmp_path / "artifacts"
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model: {}\n", encoding="utf-8")
    spec = RunSpec(
        dataset="ToyMNIST", modality="2d_image", task_type="multiclass", num_classes=3,
        surrogate="resnet18", victim="resnet18", method="datacook", seed=0,
        epsilon_pixel=8.0, epsilon=8 / 255, run_name="Toy-run", run_dir=run_dir,
        model_config=model_config, data_path=tmp_path, test_count=2,
    )
    for path in spec.predictions.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    spec.pseudo_labels.parent.mkdir(parents=True, exist_ok=True)
    spec.pseudo_labels.touch()
    spec.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    spec.checkpoint.touch()
    spec.test_perturbation.touch()
    raw_path = artifact_root / spec.dataset / "raw_test.npz"
    raw_val_path = artifact_root / spec.dataset / "raw_val.npz"
    cooked_path = artifact_root / spec.run_name / "cooked_test.npz"
    raw_path.parent.mkdir(parents=True)
    cooked_path.parent.mkdir(parents=True)
    arrays = {"images": np.zeros((2, 3, 2, 2), np.float32), "labels": np.array([0, 1])}
    np.savez_compressed(raw_path, **arrays)
    validation_arrays = {
        "images": np.zeros((60, 3, 2, 2), np.float32),
        "labels": np.tile(np.arange(3), 20),
    }
    np.savez_compressed(raw_val_path, **validation_arrays)
    np.savez_compressed(cooked_path, **arrays)

    counts = generate_configs(
        [spec], artifact_root, config_dir,
        attacks_epsilon=8, runtime_seed=0, runtime_epsilon=8,
    )
    assert counts == {
        "pseudo": 1, "image_quality": 1, "runtime": 1,
        "attacks": 1, "finetune_attacks": 1, "finetune_budget_attacks": 1,
    }
    attacks = yaml.safe_load((config_dir / "attacks.yaml").read_text(encoding="utf-8"))
    assert len(attacks["experiments"][0]["attacks"]) == 11
    finetune = yaml.safe_load((config_dir / "attacks_finetune.yaml").read_text(encoding="utf-8"))
    assert [attack["type"] for attack in finetune["experiments"][0]["attacks"]] == [
        "finetune", "finetune", "robust_finetune", "robust_finetune",
    ]
    assert finetune["output"].endswith("adaptive_finetune_results.csv")
    assert finetune["combined_output"].endswith("adaptive_attack_results_all.csv")
    budget = yaml.safe_load((config_dir / "attacks_finetune_budget.yaml").read_text(encoding="utf-8"))
    assert [(attack["type"], attack["subset_size"], attack["epochs"]) for attack in budget["experiments"][0]["attacks"]] == [
        ("finetune", 25, 10), ("robust_finetune", 25, 10),
        ("finetune", 50, 10), ("robust_finetune", 50, 10),
    ]
    final_analysis = yaml.safe_load((config_dir / "final_analysis.yaml").read_text(encoding="utf-8"))
    assert final_analysis["full_subset_sizes"] == {"ToyMNIST": 60}
    master = yaml.safe_load((config_dir.parent / "generated.yaml").read_text(encoding="utf-8"))
    assert master["tasks"]["pseudo"]["enabled"] is True
    assert master["tasks"]["finetune_attacks"]["enabled"] is False
    assert master["tasks"]["finetune_budget_attacks"]["enabled"] is False
    assert master["tasks"]["epsilon"]["enabled"] is False

    suffix_dir = tmp_path / "configs" / "revision" / "extension" / "generated"
    generate_configs(
        [spec], artifact_root, suffix_dir,
        attacks_epsilon=8, runtime_seed=0, runtime_epsilon=8,
        output_suffix="seed 3-7/eps8", finetune_epochs=(10,), budget_sizes=(25,),
    )
    suffix_attacks = yaml.safe_load((suffix_dir / "attacks_finetune.yaml").read_text(encoding="utf-8"))
    assert suffix_attacks["output"].endswith("adaptive_finetune_results_seed-3-7-eps8.csv")
    assert suffix_attacks["merge_with"].endswith("adaptive_attack_results_seed-3-7-eps8.csv")
    assert [attack["epochs"] for attack in suffix_attacks["experiments"][0]["attacks"]] == [10, 10]
    suffix_budget = yaml.safe_load((suffix_dir / "attacks_finetune_budget.yaml").read_text(encoding="utf-8"))
    assert {attack["subset_size"] for attack in suffix_budget["experiments"][0]["attacks"]} == {25}

    selected_dir = tmp_path / "configs" / "revision" / "3d" / "generated"
    volume_spec = replace(spec, modality="3d_volume")
    selected_counts = generate_configs(
        [volume_spec], artifact_root, selected_dir,
        attacks_epsilon=8, runtime_seed=0, runtime_epsilon=8,
        tasks={"pseudo", "runtime"},
        result_root="results/revision_pipeline_3d",
    )
    assert selected_counts == {
        "pseudo": 1, "image_quality": 0, "runtime": 1,
        "attacks": 0, "finetune_attacks": 0, "finetune_budget_attacks": 0,
    }
    pseudo = yaml.safe_load((selected_dir / "pseudo.yaml").read_text(encoding="utf-8"))
    runtime = yaml.safe_load((selected_dir / "runtime.yaml").read_text(encoding="utf-8"))
    assert pseudo["outputs"]["quality"] == (
        "results/revision_pipeline_3d/posthoc/pseudo_label_quality.csv"
    )
    assert runtime["output"] == "results/revision_pipeline_3d/posthoc/runtime_summary.csv"
    assert runtime["include_average"] is True
    assert runtime["experiments"][0]["batch_size"] == 16
    selected_master = yaml.safe_load(
        (selected_dir.parent / "generated.yaml").read_text(encoding="utf-8")
    )
    assert selected_master["tasks"]["pseudo"]["enabled"] is True
    assert selected_master["tasks"]["runtime"]["enabled"] is True
    assert selected_master["tasks"]["image_quality"]["enabled"] is False
    assert selected_master["tasks"]["attacks"]["enabled"] is False


def test_cooked_index_requires_global_suffix(tmp_path):
    assert _cooked_index(tmp_path / "17_123.png") == 123
    try:
        _cooked_index(tmp_path / "sample.png")
    except ValueError as error:
        assert "global sample index" in str(error)
    else:
        raise AssertionError("Expected filename validation to fail.")


class _NullLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass
