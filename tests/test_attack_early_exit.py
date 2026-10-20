from __future__ import annotations

import time

import torch

from miner26.attack import AttackConfig, Attacker
from miner26.scoring import QUANT, estimate_score, rmse_for_unit_changes
from tests.conftest import TinyNet


class AliasNet(TinyNet):
    """TinyNet that always predicts 264 ("Cardigan", the corgi), which the validator resolves to 474."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = super().forward(x)
        logits[:, 264] += 1000.0
        return logits


def test_alias_image_exits_at_the_ceiling(clean_u8: torch.Tensor) -> None:
    attacker = Attacker(AliasNet().eval(), torch.device("cpu"), AttackConfig(search_amp=False, seed=3))
    t0 = time.time()
    result = attacker.run(clean_u8, deadline=t0 + 10.0)
    elapsed = time.time() - t0
    assert result.info["alias_target"] and result.true_index == 474 and result.clean_index == 264
    assert result.info["early_exit"]
    assert result.steps == 1
    assert elapsed < 5.0, "the attack must stop as soon as nothing can improve"
    assert result.flipped and result.margin >= 10.0
    # Only the novelty seed is changed, and the estimate is the best reachable for it.
    px = attacker.cfg.novelty_pixels
    assert result.nonzero == px and result.changed_pixels == px
    best = estimate_score(linf=QUANT, rmse=rmse_for_unit_changes(px, clean_u8.numel()), margin=10.0,
                          changed_pixels=px, flipped=True)
    assert abs(result.est_score - best) < 1e-9
    assert int((result.adv_u8.int() - clean_u8.int()).abs().max()) == 1


def test_ordinary_image_runs_to_the_deadline(clean_u8: torch.Tensor) -> None:
    attacker = Attacker(TinyNet().eval(), torch.device("cpu"), AttackConfig(search_amp=False, safety_margin=0.1))
    result = attacker.run(clean_u8, deadline=time.time() + 1.0)
    assert not result.info["early_exit"]
    assert result.steps > 1
