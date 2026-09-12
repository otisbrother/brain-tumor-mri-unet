import torch

from src.models.unet import UNet


def test_unet_output_shape_240() -> None:
    model = UNet(in_channels=1, out_channels=1, base_channels=8)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 1, 240, 240))
    assert output.shape == (2, 1, 240, 240)


def test_unet_supports_configurable_modalities_and_odd_size() -> None:
    model = UNet(in_channels=4, out_channels=1, base_channels=4)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 4, 65, 67))
    assert output.shape == (1, 1, 65, 67)

