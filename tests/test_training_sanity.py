import copy

import torch

from src.inference.predict import load_model_checkpoint
from src.models.unet import UNet
from src.training.losses import BCEDiceLoss
from src.training.trainer import load_training_checkpoint, save_checkpoint


def test_unet_can_reduce_loss_and_checkpoint_round_trip(tmp_path) -> None:
    """Synthetic test data only — this is not a medical performance result."""
    torch.manual_seed(42)
    images = torch.zeros(2, 1, 32, 32)
    targets = torch.zeros_like(images)
    images[:, :, 8:24, 8:24] = 1.0
    targets[:, :, 8:24, 8:24] = 1.0
    model = UNet(1, 1, base_channels=4)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.train()
    with torch.no_grad():
        initial_loss = float(criterion(model(images), targets))
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(criterion(model(images), targets))
    assert final_loss < initial_loss

    config = {
        "project": {"name": "synthetic-software-test", "seed": 42},
        "data": {"modalities": ["flair"], "target_size": 32},
        "model": {"name": "unet", "in_channels": 1, "out_channels": 1, "base_channels": 4, "bilinear": True},
        "training": {},
        "inference": {"threshold": 0.5},
        "paths": {},
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    path = save_checkpoint(tmp_path / "model.pt", 12, model, optimizer, scheduler, 0.5, copy.deepcopy(config), 0.5)
    restored, checkpoint = load_model_checkpoint(config, path, torch.device("cpu"))
    with torch.no_grad():
        torch.testing.assert_close(restored(images), model.eval()(images))
    assert checkpoint["epoch"] == 12
    assert checkpoint["optimizer_state_dict"]

    resumed_model = UNet(1, 1, base_channels=4)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(resumed_optimizer)
    resumed = load_training_checkpoint(
        path,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        scaler=None,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        torch.testing.assert_close(resumed_model.eval()(images), model.eval()(images))
    assert resumed["epoch"] == 12


def test_data_parallel_checkpoint_is_saved_without_module_prefix(tmp_path) -> None:
    model = torch.nn.DataParallel(UNet(1, 1, base_channels=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    config = {
        "project": {"name": "checkpoint-test", "seed": 42},
        "data": {"modalities": ["flair"], "target_size": 32},
        "model": {
            "name": "unet",
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 4,
            "bilinear": True,
        },
        "training": {"multi_gpu": True},
        "inference": {"threshold": 0.5},
        "paths": {},
    }
    path = save_checkpoint(
        tmp_path / "parallel.pt",
        1,
        model,
        optimizer,
        scheduler,
        0.1,
        config,
        0.5,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert not any(key.startswith("module.") for key in checkpoint["model_state_dict"])
