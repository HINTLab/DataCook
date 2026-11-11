from pathlib import Path


def test_samplewise_eval_uses_defined_eval_target():
    source = Path("perturbation.py").read_text(encoding="utf-8")
    function = source.split("def samplewise_perturbation_eval", 1)[1]
    function = function.split("def main", 1)[0]
    assert "_ensure_labels_shape(eval_target, labels)" in function
    assert "_ensure_labels_shape(loader_key, labels)" not in function


def test_surrogate_resume_infers_device_from_model():
    source = Path("perturbation.py").read_text(encoding="utf-8")
    train_function = source.split("def train(args, config", 1)[1]
    train_function = train_function.split("def check_ssim", 1)[0]
    assert "model_device = next(model.parameters()).device" in train_function
    assert "map_location=model_device" in train_function
