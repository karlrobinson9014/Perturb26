"""Perturb26 production miner.

Reuses upstream ``PerturbMiner`` for wallet, subtensor, task polling, storage
upload and signed submission, and replaces only the attack and its timing.

    # live mining (normally via pm2: harness/ecosystem.config.js app perturb26-miner)
    python -m miner26.neuron --netuid <N> --wallet.name miner --wallet.hotkey default
    # attack the current live task, score it locally, submit nothing (no wallet needed)
    python -m miner26.neuron --dry-run

Timing: the validator reads ``GET /submits`` 40 s after the task boundary and
takes whatever is there on the first non-empty poll, so the response must be
submitted before ``boundary + SUBMIT_BY_SECONDS``. The attack deadline is that
moment minus a measured upload/submit reserve (EMA).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import torch

from miner26._upstream import ensure_bittensor_importable
from miner26.attack import AttackConfig, Attacker
from miner26.imaging import b64_to_u8, u8_to_b64

logger = logging.getLogger("miner26")

SUBMIT_BY_SECONDS = float(os.getenv("PERTURB26_SUBMIT_BY_SECONDS", "34"))
UPLOAD_RESERVE_SECONDS = float(os.getenv("PERTURB26_UPLOAD_RESERVE_SECONDS", "5"))


def _attack_config_from_env() -> AttackConfig:
    raw = os.getenv("PERTURB26_ATTACK_CONFIG", "").strip()
    return AttackConfig(**json.loads(raw)) if raw else AttackConfig()


def _task_boundary(now: float, cadence: int) -> float:
    return float(int(now // max(1, cadence)) * max(1, cadence))


def dry_run(budget: float) -> int:
    ensure_bittensor_importable()
    from bench.official import OfficialScorer
    from perturbnet import constants as C
    from perturbnet.api_client import get_current_task
    from perturbnet.image_io import image_url_to_b64
    from perturbnet.model import load_efficientnet_v2_l

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_efficientnet_v2_l(device)
    attacker = Attacker(model, device, _attack_config_from_env())
    task = get_current_task(base_url=C.PERTURB_API_BASE_URL, timeout_seconds=C.PERTURB_API_TIMEOUT_SECONDS)
    if task is None:
        logger.error("no current task")
        return 1
    clean_b64 = image_url_to_b64(task.image_url, timeout_seconds=C.PERTURB_API_TIMEOUT_SECONDS)
    t0 = time.time()
    result = attacker.run(b64_to_u8(clean_b64), deadline=t0 + budget)
    adv_b64 = u8_to_b64(result.adv_u8)
    scored = OfficialScorer(model, device).score(task.task_id, clean_b64, adv_b64)
    logger.info(
        f"dry-run task={task.task_id} official={scored['official']:.5f} est={result.est_score:.5f} "
        f"margin={result.margin:.3f} nonzero={result.nonzero} steps={result.steps} elapsed={time.time() - t0:.1f}s "
        f"reasons={scored['reasons']}"
    )
    return 0


def run_miner() -> int:
    if not ensure_bittensor_importable():
        raise SystemExit("bittensor is required for live mining (pip install -r upstream/requirements.txt)")
    from neurons.miner import PerturbMiner, build_config
    from perturbnet import constants as C
    from perturbnet.api_client import submit_miner_response
    from perturbnet.image_io import image_pixel_hash, image_url_to_b64

    class Perturb26Miner(PerturbMiner):
        def __init__(self, config) -> None:
            super().__init__(config)
            self.attacker = Attacker(self.model, self.device, _attack_config_from_env())
            self.upload_reserve = UPLOAD_RESERVE_SECONDS

        def _process_task(self, *, task_id: str, image_url: str) -> None:
            perturb = self.config.perturb
            cadence = int(getattr(perturb, "task_cadence_seconds", C.TASK_CADENCE_SECONDS))
            timeout = float(getattr(perturb, "api_timeout_seconds", C.PERTURB_API_TIMEOUT_SECONDS))
            started = time.time()
            boundary = _task_boundary(started, cadence)
            clean_b64 = image_url_to_b64(image_url, timeout_seconds=timeout)
            deadline = boundary + SUBMIT_BY_SECONDS - self.upload_reserve
            result = self.attacker.run(b64_to_u8(clean_b64), deadline=deadline)
            adv_b64 = u8_to_b64(result.adv_u8)

            upload_started = time.time()
            response_url = self._upload_response(task_id=task_id, perturbed_image_b64=adv_b64)
            submit_response = submit_miner_response(
                base_url=str(getattr(perturb, "api_base_url", C.PERTURB_API_BASE_URL)),
                wallet=self.wallet,
                image_url=response_url,
                image_hash=image_pixel_hash(adv_b64),
                timeout_seconds=timeout,
            )
            upload_seconds = time.time() - upload_started
            self.upload_reserve = max(2.0, 0.7 * self.upload_reserve + 0.3 * (upload_seconds + 1.0))
            if not self._submission_succeeded(submit_response):
                raise RuntimeError(f"Response submission failed task={task_id} api_response={submit_response}")
            logger.info(
                f"Submitted task={task_id} est={result.est_score:.5f} margin={result.margin:.3f} "
                f"nonzero={result.nonzero} steps={result.steps} attack={result.elapsed:.1f}s "
                f"upload={upload_seconds:.1f}s since_boundary={time.time() - boundary:.1f}s url={response_url}"
            )

    miner = Perturb26Miner(config=build_config())
    miner.run()
    return 0


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dry-run", action="store_true")
    pre.add_argument("--budget", type=float, default=22.0)
    ours, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0], *rest]  # upstream build_config parses the remaining flags
    if ours.dry_run:
        return dry_run(ours.budget)
    return run_miner()


if __name__ == "__main__":
    raise SystemExit(main())
