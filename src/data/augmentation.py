"""Synchronized, conservative 2D MRI and mask augmentation."""

from __future__ import annotations

import math
import random
from typing import Any

import torch
import torch.nn.functional as functional


class SynchronizedAugmentation:
    """Apply the same flip/rotation to MRI channels and segmentation mask."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.flip_probability = float(config.get("horizontal_flip_probability", 0.5))
        self.rotation_degrees = float(config.get("rotation_degrees", 10.0))
        self.intensity_scale_limit = float(config.get("intensity_scale_limit", 0.1))
        self.noise_std = float(config.get("gaussian_noise_std", 0.02))

    @staticmethod
    def _rotate(tensor: torch.Tensor, angle_degrees: float, mode: str) -> torch.Tensor:
        angle = math.radians(angle_degrees)
        theta = tensor.new_tensor(
            [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0]]
        ).unsqueeze(0)
        batched = tensor.unsqueeze(0)
        grid = functional.affine_grid(theta, batched.shape, align_corners=False)
        return functional.grid_sample(
            batched, grid, mode=mode, padding_mode="zeros", align_corners=False
        ).squeeze(0)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.flip_probability:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = self._rotate(image, angle, mode="bilinear")
            mask = self._rotate(mask, angle, mode="nearest")
        if self.intensity_scale_limit > 0:
            scale = random.uniform(1 - self.intensity_scale_limit, 1 + self.intensity_scale_limit)
            image = image * scale
        if self.noise_std > 0:
            brain = (image != 0).any(dim=0, keepdim=True)
            image = image + torch.randn_like(image) * self.noise_std * brain
        return image.float(), (mask > 0.5).float()

