from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from run_tabular_epsilon_ablation import (
    TabularNet,
    cook_tabular_inputs,
    load_tabular_split,
    run_ablation,
)


FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def test_tabular_cooker_respects_mask_and_linf_budget():
    torch.manual_seed(0)
    model = TabularNet(8, dropout=0)
    inputs = torch.randn(12, 8)
    cooked, elapsed = cook_tabular_inputs(
        model,
        inputs,
        epsilon=0.2,
        attackable_indices=[5, 6],
        x_min=inputs.min(dim=0).values,
        x_max=inputs.max(dim=0).values,
        num_steps=2,
        step_size_ratio=0.5,
        device=torch.device("cpu"),
    )
    delta = cooked - inputs
    assert elapsed >= 0
    assert float(delta.abs().max()) <= 0.2 + 1e-6
    np.testing.assert_allclose(delta[:, :5].numpy(), 0, atol=1e-7)
    np.testing.assert_allclose(delta[:, 7:].numpy(), 0, atol=1e-7)


def test_multilabel_tabular_cooker_respects_budget():
    torch.manual_seed(0)
    model = TabularNet(5, output_dim=3, dropout=0)
    inputs = torch.randn(12, 5)
    cooked, _ = cook_tabular_inputs(
        model,
        inputs,
        epsilon=0.15,
        attackable_indices=[1, 3],
        x_min=inputs.min(dim=0).values,
        x_max=inputs.max(dim=0).values,
        num_steps=2,
        step_size_ratio=0.5,
        device=torch.device("cpu"),
        task_type="multilabel",
    )
    delta = cooked - inputs
    assert float(delta.abs().max()) <= 0.15 + 1e-6
    np.testing.assert_allclose(delta[:, [0, 2, 4]].numpy(), 0, atol=1e-7)


def test_tabular_cooker_supports_absolute_step_size():
    torch.manual_seed(0)
    model = TabularNet(8, dropout=0)
    inputs = torch.randn(12, 8)
    cooked, _ = cook_tabular_inputs(
        model,
        inputs,
        epsilon=5.0,
        attackable_indices=[5, 6],
        x_min=torch.full((8,), -100.0),
        x_max=torch.full((8,), 100.0),
        num_steps=1,
        step_size=0.05,
        device=torch.device("cpu"),
    )
    delta = cooked - inputs
    assert float(delta.abs().max()) <= 0.05 + 1e-6
    assert float(delta[:, [5, 6]].abs().max()) >= 0.05 - 1e-6
    np.testing.assert_allclose(delta[:, :5].numpy(), 0, atol=1e-7)
    np.testing.assert_allclose(delta[:, 7:].numpy(), 0, atol=1e-7)


def test_tabular_loader_supports_categorical_multiclass(tmp_path):
    frame = pd.DataFrame({
        "Age": np.arange(30),
        "Sex": np.tile(["m", "f"], 15),
        "Category": np.repeat(["healthy", "a", "b"], 10),
    })
    path = tmp_path / "multiclass.csv"
    frame.to_csv(path, index=False)
    split = load_tabular_split(
        path,
        features=["Age", "Sex"],
        targets=["Category"],
        task_type="multiclass",
        seed=0,
        test_size=0.2,
        categorical_mappings={"Sex": {"m": 0, "f": 1}},
        target_mapping={"healthy": 0, "a": 1, "b": 2},
    )
    x_train, y_train, x_test, y_test, _, _ = split
    assert x_train.shape == (24, 2)
    assert x_test.shape == (6, 2)
    assert y_train.dtype == torch.int64
    assert set(torch.cat((y_train, y_test)).tolist()) == {0, 1, 2}


def test_small_tabular_ablation_exports_unified_results(tmp_path):
    rng = np.random.default_rng(7)
    rows = 80
    frame = pd.DataFrame({
        feature: rng.normal(size=rows)
        for feature in FEATURES
    })
    frame["Outcome"] = np.tile([0, 1], rows // 2)
    data_path = tmp_path / "diabetes.csv"
    frame.to_csv(data_path, index=False)
    artifact_root = tmp_path / "artifacts"
    result_root = tmp_path / "results"
    config = {
        "dataset": {
            "name": "Diabetes",
            "data_path": str(data_path),
            "features": FEATURES,
            "target": "Outcome",
            "test_size": 0.25,
        },
        "model": {"hidden_dims": [8], "dropout": 0},
        "experiment": {
            "seeds": [0],
            "epsilons": [0.1],
            "attackable_features": ["BMI", "DiabetesPedigreeFunction"],
            "num_steps": 1,
            "step_size": 0.05,
            "baseline_epochs": 2,
            "protected_epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.01,
        },
        "outputs": {
            "artifact_root": str(artifact_root),
            "result_root": str(result_root),
        },
    }
    args = SimpleNamespace(
        data_path=None,
        seeds=None,
        epsilons=None,
        device="cpu",
        force=False,
        dry_run=False,
    )

    outputs = run_ablation(config, tmp_path, args)

    assert set(outputs) == {"main_metrics", "cp_pp", "epsilon", "runtime"}
    assert all(path.is_file() for path in outputs.values())
    epsilon = pd.read_csv(outputs["epsilon"])
    assert set(epsilon["metric"]) >= {"acc", "auc", "balanced_acc"}
    assert set(epsilon["epsilon_scale"]) == {"standardized_feature_linf"}
    assert set(epsilon["num_steps"]) == {1}
    assert set(epsilon["step_size"]) == {0.05}
    assert epsilon["step_size_ratio"].isna().all()
    runtime = pd.read_csv(outputs["runtime"])
    assert len(runtime) == 1
    assert runtime.loc[0, "num_samples"] == 20
    assert runtime.loc[0, "cooking_time_per_sample"] >= 0
    assert runtime.loc[0, "overhead_ratio"] > 0
    with np.load(
        artifact_root
        / "Diabetes-datacook-sur=mlp-vic=mlp-seed=0-eps=0.1"
        / "predictions"
        / "authorized_cooked.npz"
    ) as artifact:
        assert artifact["y_true"].shape == (20,)
        assert artifact["y_prob"].shape == (20, 2)


def test_small_multilabel_ablation_exports_multilabel_predictions(tmp_path):
    rng = np.random.default_rng(11)
    rows = 60
    features = ["F0", "F1", "F2", "F3"]
    frame = pd.DataFrame({
        feature: rng.normal(size=rows)
        for feature in features
    })
    frame["L0"] = np.tile([0, 1], rows // 2)
    frame["L1"] = np.tile([0, 0, 1, 1], rows // 4)
    frame["L2"] = np.tile([0, 0, 0, 1, 1, 1], rows // 6)
    data_path = tmp_path / "multilabel.csv"
    frame.to_csv(data_path, index=False)
    artifact_root = tmp_path / "artifacts"
    result_root = tmp_path / "results"
    config = {
        "dataset": {
            "name": "ToyMultilabel",
            "task_type": "multilabel",
            "data_path": str(data_path),
            "features": features,
            "targets": ["L0", "L1", "L2"],
            "test_size": 0.25,
        },
        "model": {"hidden_dims": [8], "dropout": 0},
        "experiment": {
            "seeds": [0],
            "epsilons": [0.1],
            "attackable_features": ["F1", "F3"],
            "num_steps": 1,
            "step_size_ratio": 0.5,
            "baseline_epochs": 2,
            "protected_epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.01,
            "class_balance": True,
        },
        "outputs": {
            "artifact_root": str(artifact_root),
            "result_root": str(result_root),
        },
    }
    args = SimpleNamespace(
        data_path=None,
        seeds=None,
        epsilons=None,
        device="cpu",
        force=False,
        dry_run=False,
    )

    outputs = run_ablation(config, tmp_path, args)

    epsilon = pd.read_csv(outputs["epsilon"])
    assert set(epsilon["metric"]) >= {"subset_acc", "macro_auc", "micro_f1"}
    assert set(epsilon["task_type"]) == {"multilabel"}
    with np.load(
        artifact_root
        / "ToyMultilabel-datacook-sur=mlp-vic=mlp-seed=0-eps=0.1"
        / "predictions"
        / "authorized_cooked.npz"
    ) as artifact:
        assert artifact["y_true"].shape == (15, 3)
        assert artifact["y_pred"].shape == (15, 3)
        assert artifact["y_prob"].shape == (15, 3)
