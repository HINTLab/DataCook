from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from evaluator import Evaluator
from toolbox import PerturbationTool
from volume_io import save_cooked_array_npz


def test_multilabel_cooked_npz_preserves_label_matrix(tmp_path):
    output = tmp_path / "perturb_data.npz"
    images = np.random.default_rng(0).random((3, 3, 8, 8), dtype=np.float32)
    labels = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.int64)
    save_cooked_array_npz(output, images, labels)

    # dataset.py is available in the project/server environment. Lightweight
    # analysis environments may omit its mlconfig runtime dependency.
    pytest.importorskip("mlconfig")
    from dataset import Data3DFolderWithLabel

    dataset = Data3DFolderWithLabel(output)
    image, label, prediction_label = dataset[1]
    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert label.tolist() == [0, 1]
    assert prediction_label.tolist() == [0, 1]


def _multilabel_config():
    return SimpleNamespace(
        log_frequency=100,
        num_classes=2,
        task_type="multilabel",
        grad_clip=5.0,
    )


def test_bce_trainer_uses_multilabel_targets():
    pytest.importorskip("mlconfig")
    from trainer import Trainer

    config = _multilabel_config()
    criterion = torch.nn.BCEWithLogitsLoss()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([[1, 0], [0, 1]])
    trainer = Trainer(criterion, {}, None, config, "ChestMNIST")
    payload = trainer.train_batch(images, labels, model, optimizer)
    assert 0.0 <= payload["acc"] <= 1.0


def test_evaluator_exports_sigmoid_multilabel_predictions():
    config = _multilabel_config()
    model = torch.nn.Linear(2, 2)
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([[1, 0], [0, 1]])
    evaluator = Evaluator({}, None, config, "ChestMNIST")
    eval_payload = evaluator.eval_batch(images, labels, model)
    assert 0.0 <= eval_payload["acc"] <= 1.0
    assert evaluator.y_true[0].shape == (2, 2)
    assert evaluator.y_prob[0].shape == (2, 2)
    assert torch.all((evaluator.y_prob[0] >= 0) & (evaluator.y_prob[0] <= 1))


def test_datacook_generates_per_label_pseudo_targets_for_bce():
    model = torch.nn.Linear(2, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    tool = PerturbationTool(epsilon=0.1, num_steps=1, step_size=0.01)
    images = torch.rand(4, 2)
    labels = torch.zeros(4, 3)
    perturb_img, delta = tool.datacook_v1(
        images,
        labels,
        model,
        optimizer,
        torch.nn.BCEWithLogitsLoss(),
        random_noise=torch.zeros_like(images),
    )
    assert perturb_img.shape == images.shape
    assert delta.shape == images.shape


def test_chestmnist_pipeline_contract():
    config = yaml.safe_load(Path("configs/pipelines/chestmnist.yaml").read_text())
    dataset = config["dataset"]
    assert dataset["task_type"] == "multilabel"
    assert dataset["num_classes"] == 14
    assert dataset["noise_shape"] == [78468, 3, 28, 28]
    assert dataset["test_noise_shape"] == [22433, 3, 28, 28]

    model_config = yaml.safe_load(
        Path("configs/MedMNIST/ChestMNIST/resnet18.yaml").read_text()
    )
    assert model_config["criterion"]["name"] == "BCEWithLogitsLoss"
    assert model_config["task_type"] == "multilabel"
