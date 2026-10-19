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
moment minus a measured upload/submit reserve (EMA). Task fetch and image
download use short per-request timeouts with quick retries (``miner26.fetch``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time

import torch

from miner26._upstream import ensure_bittensor_importable
from miner26.attack import AttackConfig, Attacker
from miner26.fetch import download_with_retries, fetch_new_task, task_epoch
from miner26.imaging import b64_to_u8, u8_to_b64

logger = logging.getLogger("miner26")

SUBMIT_BY_SECONDS = float(os.getenv("PERTURB26_SUBMIT_BY_SECONDS", "34"))
UPLOAD_RESERVE_SECONDS = float(os.getenv("PERTURB26_UPLOAD_RESERVE_SECONDS", "5"))
# Stop looking for the task when less than this much attack time would remain.
MIN_ATTACK_SECONDS = float(os.getenv("PERTURB26_MIN_ATTACK_SECONDS", "5"))


def _attack_config_from_env(hotkey: str = "") -> AttackConfig:
    raw = os.getenv("PERTURB26_ATTACK_CONFIG", "").strip()
    overrides = json.loads(raw) if raw else {}
    if hotkey and "seed" not in overrides:
        # Per-hotkey seed: identical responses from two miners are both zeroed.
        overrides["seed"] = int(hashlib.sha256(hotkey.encode("utf-8")).hexdigest()[:8], 16)
    return AttackConfig(**overrides)


def _task_boundary(now: float, cadence: int, early_tolerance: float = 5.0) -> float:
    """Boundary of the task fetched at ``now``; tolerates seeing it slightly before the boundary."""
    cadence = max(1, cadence)
    return float(int((now + early_tolerance) // cadence) * cadence)


def dry_run(budget: float) -> int:
    ensure_bittensor_importable()
    from bench.official import OfficialScorer
    from perturbnet import constants as C
    from perturbnet.api_client import get_current_task
    from perturbnet.image_io import image_url_to_b64
    from perturbnet.model import load_efficientnet_v2_l

    # force: bittensor installs its own root handlers, which made basicConfig a no-op (dry run printed nothing)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_efficientnet_v2_l(device)
    attacker = Attacker(model, device, _attack_config_from_env())
    task = fetch_new_task(
        get_current_task, base_url=C.PERTURB_API_BASE_URL, last_task_id=None, boundary=0.0, deadline=time.time() + 15
    )
    if task is None:
        logger.error("no current task")
        return 1
    clean_b64 = download_with_retries(image_url_to_b64, task.image_url, deadline=time.time() + 15)
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
    from perturbnet.api_client import get_current_task
    from perturbnet.image_io import image_pixel_hash, image_url_to_b64

    class Perturb26Miner(PerturbMiner):
        def __init__(self, config) -> None:
            super().__init__(config)
            hotkey = str(getattr(self.wallet.hotkey, "ss58_address", ""))
            self.attacker = Attacker(self.model, self.device, _attack_config_from_env(hotkey))
            self.upload_reserve = UPLOAD_RESERVE_SECONDS

        def _api_base_url(self) -> str:
            return str(getattr(self.config.perturb, "api_base_url", C.PERTURB_API_BASE_URL))

        def _cadence(self) -> int:
            return int(getattr(self.config.perturb, "task_cadence_seconds", C.TASK_CADENCE_SECONDS))

        def _snapshot_current_task_before_boundary(self) -> None:
            # Upstream: one request with a 10 s timeout. Here: short requests until 1 s before the boundary.
            cadence = max(1, self._cadence())
            prefetch = float(getattr(self.config.perturb, "task_pre_boundary_fetch_seconds", C.TASK_PRE_BOUNDARY_FETCH_SECONDS))
            prefetch = min(max(2.0, prefetch), float(cadence))
            now = time.time()
            boundary = (int(now // cadence) + 1) * cadence
            if boundary - prefetch > now:
                time.sleep(boundary - prefetch - now)
            task = fetch_new_task(
                get_current_task, base_url=self._api_base_url(), last_task_id=None, boundary=0.0, deadline=boundary - 1.0
            )
            if task is not None:
                self.last_processed_task_id = task.task_id
                logger.info(f"Pre-boundary baseline task_id={self.last_processed_task_id}")

        def _fetch_new_task_at_boundary(self):
            boundary = _task_boundary(time.time(), self._cadence())
            return fetch_new_task(
                get_current_task,
                base_url=self._api_base_url(),
                last_task_id=self.last_processed_task_id,
                boundary=boundary,
                deadline=boundary + SUBMIT_BY_SECONDS - self.upload_reserve - MIN_ATTACK_SECONDS,
            )

        def _process_task(self, *, task_id: str, image_url: str) -> None:
            perturb = self.config.perturb
            timeout = float(getattr(perturb, "api_timeout_seconds", C.PERTURB_API_TIMEOUT_SECONDS))
            started = time.time()
            epoch = task_epoch(task_id)
            boundary = float(epoch) if epoch is not None and abs(started - epoch) < 60 else _task_boundary(started, self._cadence())
            self.last_processed_task_id = task_id
            deadline = boundary + SUBMIT_BY_SECONDS - self.upload_reserve
            clean_b64 = download_with_retries(image_url_to_b64, image_url, deadline=deadline - MIN_ATTACK_SECONDS)
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
