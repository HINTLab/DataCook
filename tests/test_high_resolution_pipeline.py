from pathlib import Path

import yaml

from prepare_revision_posthoc import load_run_specs
from run_datacook_pipeline import _main_command
from run_datacook_suite import load_pipeline_configs


def test_224_suite_uses_three_datasets_and_safe_defaults():
    suite = Path("configs/pipelines/medmnist_224_three.yaml").resolve()
    configs = load_pipeline_configs(suite)
    assert [path.name for path in configs] == [
        "breastmnist224.yaml",
        "pneumoniamnist224.yaml",
        "retinamnist224.yaml",
    ]
    expected_counts = {
        "BreastMNIST": (546, 156),
        "PneumoniaMNIST": (4708, 624),
        "RetinaMNIST": (1080, 400),
    }
    for path in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        dataset = config["dataset"]
        train_count, test_count = expected_counts[dataset["name"]]
        assert dataset["high_resolution"] is True
        assert dataset["modality"] == "high_resolution"
        assert dataset["noise_shape"] == [train_count, 3, 224, 224]
        assert dataset["test_noise_shape"] == [test_count, 3, 224, 224]
        assert config["models"] == {
            "surrogate": "resnet18_224",
            "victim": "resnet18_224",
        }
        assert config["seeds"] == [0, 1, 2]
        assert config["epsilons"] == [8]


def test_224_flag_is_forwarded_to_training_command(tmp_path):
    command = _main_command(
        "python",
        model="resnet18_224",
        config_path=tmp_path,
        exp_name=tmp_path / "exp",
        dataset_name="BreastMNIST",
        train_type="BreastMNIST",
        train_path=tmp_path,
        test_type="BreastMNIST",
        test_path=tmp_path,
        recover_type="BreastMNIST",
        recover_path=tmp_path,
        prediction_dir=tmp_path / "predictions",
        role="clean",
        seed=0,
        train_batch_size=8,
        eval_batch_size=8,
        workers=0,
        checkpoint=tmp_path / "missing.pth",
        high_resolution=True,
    )
    assert "--high_resolution" in command


def test_posthoc_specs_keep_224_resolution_metadata():
    specs = load_run_specs(Path("configs/pipelines/medmnist_224_three.yaml").resolve())
    assert len(specs) == 9
    assert all(spec.high_resolution for spec in specs)
    assert {spec.modality for spec in specs} == {"high_resolution"}
