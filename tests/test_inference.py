import numpy as np
import torch

from src.inference.predict import predict_volume


class CenterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = torch.full((inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]), -20.0, device=inputs.device)
        logits[:, :, 2:-2, 2:-2] = 20.0 + self.anchor
        return logits


def test_whole_volume_inference_restores_original_shape() -> None:
    config = {
        "data": {"target_size": 16},
        "inference": {"threshold": 0.5, "batch_size": 2, "minimum_positive_pixels": 1},
        "postprocessing": {"enabled": False, "min_component_pixels": 20},
    }
    volume = np.ones((1, 12, 10, 3), dtype=np.float32)
    result = predict_volume(CenterModel(), volume, (0.5, 0.5, 2.0), config, torch.device("cpu"))
    assert result["mask"].shape == (12, 10, 3)
    assert result["probabilities"].shape == volume.shape[1:]
    assert result["detected"]
    assert result["tumor_voxels"] > 0

