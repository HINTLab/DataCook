from pathlib import Path

import yaml

from run_datacook_pipeline import _discover_run_results
from run_datacook_suite import load_pipeline_configs


def test_three_dataset_suite_configs_are_valid():
    suite = Path("configs/pipelines/medmnist_three.yaml").resolve()
    configs = load_pipeline_configs(suite)
    assert [path.name for path in configs] == [
        "bloodmnist.yaml",
        "breastmnist.yaml",
        "retinamnist.yaml",
    ]
    expected = {
        "BloodMNIST": (8, "multiclass", [11959, 3, 28, 28], [3421, 3, 28, 28]),
        "BreastMNIST": (2, "binary", [546, 3, 28, 28], [156, 3, 28, 28]),
        "RetinaMNIST": (5, "multiclass", [1080, 3, 28, 28], [400, 3, 28, 28]),
    }
    for path in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        dataset = config["dataset"]
        assert (
            dataset["num_classes"],
            dataset["task_type"],
            dataset["noise_shape"],
            dataset["test_noise_shape"],
        ) == expected[dataset["name"]]
        assert (Path(dataset["config_path"]) / "resnet18.yaml").is_file()


def test_additional_suite_contains_only_new_datasets():
    suite = Path("configs/pipelines/medmnist_additional_two.yaml").resolve()
    assert [path.name for path in load_pipeline_configs(suite)] == [
        "breastmnist.yaml",
        "retinamnist.yaml",
    ]


def test_result_discovery_combines_dataset_run_directories(tmp_path):
    expected = []
    for run_name in ("BloodMNIST-seed=0", "BreastMNIST-seed=0", "RetinaMNIST-seed=0"):
        path = tmp_path / run_name / "metrics" / "main_metrics.csv"
        path.parent.mkdir(parents=True)
        path.write_text("dataset,metric,value\n", encoding="utf-8")
        expected.append(path)
    summary = tmp_path / "summary" / "main_metrics.csv"
    summary.parent.mkdir()
    summary.write_text("do,not,include\n", encoding="utf-8")
    assert _discover_run_results(tmp_path, "main_metrics.csv") == sorted(expected)
