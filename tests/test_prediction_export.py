from types import SimpleNamespace

import numpy as np
import torch

from evaluator import Evaluator


def test_evaluator_prediction_export_and_reset(tmp_path):
    config = SimpleNamespace(log_frequency=100, num_classes=2, task_type="multiclass")
    evaluator = Evaluator({}, None, config, "Toy")
    evaluator.y_true = [torch.tensor([0, 1])]
    evaluator.y_prob = [torch.tensor([[0.9, 0.1], [0.2, 0.8]])]
    output = tmp_path / "clean.npz"
    evaluator.save_predictions(str(output))
    with np.load(output) as artifact:
        assert artifact["y_true"].tolist() == [0, 1]
        assert artifact["y_prob"].shape == (2, 2)
    evaluator._reset_stats()
    assert evaluator.y_true == []
    assert evaluator.recover_y_prob == []
