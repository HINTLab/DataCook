import logging
from pathlib import Path

import numpy as np

from volume_io import save_cooked_volume_npz
from run_datacook_suite import load_pipeline_configs


def test_3d_cooked_splits_are_exported_as_npz(tmp_path):
    output = tmp_path / "ae" / "perturb_data.npz"
    save_cooked_volume_npz(
        output,
        np.full((2, 1, 8, 8, 8), 0.5, dtype=np.float32),
        np.asarray([[0], [1]], dtype=np.int64),
    )

    assert output.is_file()
    with np.load(output) as archive:
        assert archive["perturb_images"].shape == (2, 1, 8, 8, 8)
        np.testing.assert_array_equal(archive["perturb_labels"], [0, 1])


def test_perturbation_uses_3d_patch_and_npz_contract():
    source = Path("perturbation.py").read_text(encoding="utf-8")
    assert "_patch_noise_to_3d_img" in source
    assert "save_cooked_volume_npz(output_path, volume_images, volume_labels)" in source


def test_3d_smoke_config_uses_single_seed_and_epsilon():
    import yaml

    config = yaml.safe_load(
        Path("configs/pipelines/organmnist3d_smoke.yaml").read_text(encoding="utf-8")
    )
    assert config["dataset"]["noise_shape"] == [971, 1, 28, 28, 28]
    assert config["dataset"]["test_noise_shape"] == [610, 1, 28, 28, 28]
    assert config["models"] == {
        "surrogate": "resnet18_3d",
        "victim": "resnet18_3d",
    }
    assert config["seeds"] == [0]
    assert config["epsilons"] == [8]


def test_four_dataset_3d_suite_has_aligned_formal_settings():
    import yaml

    suite = Path("configs/pipelines/medmnist3d_four.yaml").resolve()
    paths = load_pipeline_configs(suite)
    assert [path.name for path in paths] == [
        "organmnist3d.yaml",
        "nodulemnist3d.yaml",
        "fracturemnist3d.yaml",
        "synapsemnist3d.yaml",
    ]
    expected = {
        "OrganMNIST3D": (971, 610, 11, "multiclass"),
        "NoduleMNIST3D": (1158, 310, 2, "binary"),
        "FractureMNIST3D": (1027, 240, 3, "multiclass"),
        "SynapseMNIST3D": (1230, 352, 2, "binary"),
    }
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        dataset = config["dataset"]
        train_count, test_count, classes, task = expected[dataset["name"]]
        assert dataset["noise_shape"] == [train_count, 1, 28, 28, 28]
        assert dataset["test_noise_shape"] == [test_count, 1, 28, 28, 28]
        assert dataset["num_classes"] == classes
        assert dataset["task_type"] == task
        assert config["seeds"] == [0, 1, 2]
        assert config["epsilons"] == [8, 16, 32]
        assert config["models"]["surrogate"] == "resnet18_3d"
        assert config["models"]["victim"] == "resnet18_3d"
