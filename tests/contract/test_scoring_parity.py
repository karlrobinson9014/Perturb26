"""PROTECTED: the attack's score model must equal the official verifier.

If these fail after an upstream bump, scoring rules changed: update
miner26/scoring.py, re-baseline status/best.json, and record it in notes/.
"""
from __future__ import annotations

import pytest
import torch

from bench.official import OfficialScorer
from miner26.imaging import u8_to_b64
from miner26.scoring import QUANT, estimate_score, rmse_for_unit_changes, theoretical_max, validator_true_index
from perturbnet.model import PREPROCESS
from tests.conftest import FixedLogitModel


def _sparse_q(shape, nonzero: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    q = torch.zeros(shape).flatten()
    idx = torch.randperm(q.numel(), generator=gen)[:nonzero]
    q[idx] = torch.where(torch.rand(nonzero, generator=gen) < 0.5, -1.0, 1.0)
    return q.view(shape)


@pytest.mark.parametrize("nonzero,margin", [(1, 0.5), (8, 3.0), (500, 9.9), (20000, 12.0), (360000, 2.0)])
def test_estimate_matches_official(clean_u8: torch.Tensor, nonzero: int, margin: float) -> None:
    clean_pre = PREPROCESS((clean_u8.float() / 255.0).unsqueeze(0))
    model = FixedLogitModel(clean_pre, clean_class=5, adv_class=7, adv_margin=margin)
    scorer = OfficialScorer(model, torch.device("cpu"))
    q = _sparse_q(tuple(clean_u8.shape), nonzero, seed=nonzero)
    adv = (clean_u8.float() + q).clamp(0, 255).to(torch.uint8)
    official = scorer.score("t", u8_to_b64(clean_u8), u8_to_b64(adv))
    real_q = adv.float() - clean_u8.float()
    est = estimate_score(
        linf=QUANT,
        rmse=rmse_for_unit_changes(int((real_q != 0).sum()), real_q.numel()),
        margin=margin,
        changed_pixels=int((real_q != 0).any(dim=0).sum()),
        flipped=True,
    )
    assert official["reasons"] == ["success"]
    # The validator measures deltas in float32, where (k+1)/255 - k/255 differs from
    # 1/255 by up to ~6e-8; that moves the score by a few 1e-6 at most.
    assert official["official"] == pytest.approx(est, abs=2e-5)


def test_unchanged_prediction_scores_zero(clean_u8: torch.Tensor) -> None:
    clean_pre = PREPROCESS((clean_u8.float() / 255.0).unsqueeze(0))
    model = FixedLogitModel(clean_pre, clean_class=5, adv_class=5, adv_margin=0.0)
    q = _sparse_q(tuple(clean_u8.shape), 100, seed=1)
    adv = (clean_u8.float() + q).clamp(0, 255).to(torch.uint8)
    official = OfficialScorer(model, torch.device("cpu")).score("t", u8_to_b64(clean_u8), u8_to_b64(adv))
    assert official["official"] == 0.0


def test_two_over_255_is_much_worse() -> None:
    one = estimate_score(linf=QUANT, rmse=0.0, margin=0.0, changed_pixels=8, flipped=True)
    two = estimate_score(linf=2 * QUANT, rmse=0.0, margin=0.0, changed_pixels=8, flipped=True)
    assert one - two > 0.15


def test_theoretical_max() -> None:
    assert theoretical_max() == pytest.approx(0.99303, abs=1e-5)


def test_label_alias_collision_is_modelled() -> None:
    # torchvision has both "Cardigan" (corgi, 264) and "cardigan" (sweater, 474);
    # the validator's label-string round trip maps 264 to 474.
    assert validator_true_index(264) == 474
    assert validator_true_index(1) == 1
