"""PROTECTED: invariants every attack version must keep."""
from __future__ import annotations

import time

import torch

from bench.official import OfficialScorer
from miner26.attack import AttackConfig, Attacker
from miner26.imaging import u8_to_b64
from tests.conftest import TinyNet


def test_output_is_admissible_and_estimate_matches_official(clean_u8: torch.Tensor) -> None:
    model = TinyNet().eval()
    device = torch.device("cpu")
    # TinyNet's best margin at 1/255 is ~0.6 logits, so use a small safety margin.
    attacker = Attacker(model, device, AttackConfig(search_amp=False, safety_margin=0.1))
    t0 = time.time()
    result = attacker.run(clean_u8, deadline=t0 + 6.0)
    elapsed = time.time() - t0
    assert result.flipped and result.est_score > 0.0, "a sign attack flips TinyNet at 1/255; v0 must too"

    assert result.adv_u8.dtype == torch.uint8
    assert result.adv_u8.shape == clean_u8.shape
    delta = result.adv_u8.int() - clean_u8.int()
    assert int(delta.abs().max()) <= 1, "L-inf must stay at 1/255"
    assert elapsed < 6.0 + 3.0, "attack must respect its deadline (one step of slack)"

    official = OfficialScorer(model, device).score("t", u8_to_b64(clean_u8), u8_to_b64(result.adv_u8))
    if result.est_score > 0.0:
        assert abs(official["official"] - result.est_score) < 2e-5
