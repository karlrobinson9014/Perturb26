from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import miner26  # noqa: E402,F401  (adds upstream/ to sys.path)


class FixedLogitModel(torch.nn.Module):
    """Predicts ``clean_class`` on the clean image and ``adv_class`` with a chosen margin otherwise."""

    def __init__(self, clean_pre: torch.Tensor, clean_class: int, adv_class: int, adv_margin: float) -> None:
        super().__init__()
        self.register_buffer("clean_pre", clean_pre)
        self.clean_class, self.adv_class, self.adv_margin = clean_class, adv_class, adv_margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], 1000)
        for i in range(x.shape[0]):
            if torch.equal(x[i], self.clean_pre[0]):
                logits[i, self.clean_class] = 5.0
            else:
                logits[i, self.clean_class] = 1.0
                logits[i, self.adv_class] = 1.0 + self.adv_margin
        return logits


class TinyNet(torch.nn.Module):
    """Small random CNN with 1000 outputs: cheap stand-in for EfficientNetV2-L in attack tests."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 5, stride=4), torch.nn.ReLU(), torch.nn.Conv2d(8, 16, 5, stride=4), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(4),
        )
        self.head = torch.nn.Linear(16 * 16, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1)) * 20.0


@pytest.fixture
def clean_u8() -> torch.Tensor:
    gen = torch.Generator().manual_seed(7)
    return torch.randint(1, 255, (3, 300, 400), generator=gen, dtype=torch.uint8)
