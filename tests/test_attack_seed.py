from __future__ import annotations

import time

import torch

from miner26.attack import AttackConfig, Attacker
from tests.conftest import TinyNet


def test_different_seeds_give_different_responses(clean_u8: torch.Tensor) -> None:
    # The validator zeroes byte-identical responses from different miners.
    model = TinyNet().eval()
    outputs = []
    for seed in (1, 2):
        attacker = Attacker(model, torch.device("cpu"), AttackConfig(search_amp=False, seed=seed))
        outputs.append(attacker.run(clean_u8, deadline=time.time() + 0.5).adv_u8)
    assert not torch.equal(outputs[0], outputs[1])
    for out in outputs:
        assert int((out.int() - clean_u8.int()).abs().max()) <= 1
