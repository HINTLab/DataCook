import torch

import util


def test_checkpoint_suffix_and_resume_compatibility(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    base = tmp_path / "model"
    util.save_model(str(base), 0, model, optimizer, scheduler, ENV={"global_step": 1})
    assert (tmp_path / "model.pth").is_file()
    restored = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
    checkpoint = util.load_model(str(base), restored, restored_optimizer, restored_scheduler)
    assert checkpoint["epoch"] == 1
