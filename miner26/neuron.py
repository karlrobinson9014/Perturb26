"""Perturb26 production miner.

Reuses upstream ``PerturbMiner`` for wallet, subtensor, task polling, storage
upload and signed submission, and replaces only the attack and its timing.

    # live mining (normally via pm2: harness/ecosystem.config.js app perturb26-miner)
    python -m miner26.neuron --netuid <N> --wallet.name miner --wallet.hotkey default
    # attack the current live task, score it locally, submit nothing (no wallet needed)
    python -m miner26.neuron --dry-run
    # the next 5 live tasks with the live miner's timing, results as JSON
    python -m miner26.neuron --dry-run --rounds 5 --out results/dry-run-live.json

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


def _setup_miner_logging() -> None:
    """Own INFO handler for the ``miner26`` logger; call after bittensor is imported and configured.

    bittensor's default logging state sets every non-bittensor logger to CRITICAL (and may strip their
    handlers), which silenced all miner26 output: the 2026-10-19 five-round dry run wrote an empty log.
    """
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(getattr(h, "_miner26", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        handler._miner26 = True
        logger.addHandler(handler)


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


def _next_boundary(now: float, cadence: int) -> float:
    cadence = max(1, cadence)
    return float((int(now // cadence) + 1) * cadence)


def _summarize_rounds(rounds: list[dict]) -> dict:
    done = [r for r in rounds if r.get("official") is not None]
    scores = sorted(r["official"] for r in done)
    return {
        "rounds": len(rounds),
        "scored": len(done),
        "success": sum(1 for s in scores if s > 0.0),
        "mean_official": sum(scores) / len(scores) if scores else None,
        "min_official": scores[0] if scores else None,
        "max_done_since_boundary": max((r["done_since_boundary"] for r in done if "done_since_boundary" in r), default=None),
    }


def _write_json(path: str | None, payload: dict) -> None:
    if path:
        with open(path, "w") as f:
            json.dump(payload, f, indent=1)


def dry_run(budget: float, rounds: int = 0, out: str | None = None) -> int:
    """Attack live tasks, score them locally, submit nothing.

    ``rounds == 0``: attack the current task now with ``budget`` seconds. ``rounds > 0``: wait for the next
    ``rounds`` task boundaries and treat each new task like the live miner (fetch loop, boundary from the id
    epoch, deadline boundary + SUBMIT_BY_SECONDS - UPLOAD_RESERVE_SECONDS); ``budget`` is ignored.
    """
    ensure_bittensor_importable()
    from bench.official import OfficialScorer
    from perturbnet import constants as C
    from perturbnet.api_client import get_current_task
    from perturbnet.image_io import image_url_to_b64
    from perturbnet.model import load_efficientnet_v2_l

    _setup_miner_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_efficientnet_v2_l(device)
    attacker = Attacker(model, device, _attack_config_from_env())
    scorer = OfficialScorer(model, device)

    def attack_and_score(task, clean_b64: str, deadline: float) -> dict:
        t0 = time.time()
        result = attacker.run(b64_to_u8(clean_b64), deadline=deadline)
        attack_seconds = time.time() - t0
        scored = scorer.score(task.task_id, clean_b64, u8_to_b64(result.adv_u8))
        logger.info(
            f"dry-run task={task.task_id} official={scored['official']:.5f} est={result.est_score:.5f} "
            f"margin={result.margin:.3f} nonzero={result.nonzero} steps={result.steps} elapsed={attack_seconds:.1f}s "
            f"reasons={scored['reasons']}"
        )
        return {
            "task_id": task.task_id,
            "official": scored["official"],
            "variant_disagreement": scored["variant_disagreement"],
            "reasons": scored["reasons"],
            "est": result.est_score,
            "margin": result.margin,
            "nonzero": result.nonzero,
            "steps": result.steps,
            "attack_seconds": attack_seconds,
        }

    if rounds <= 0:
        task = fetch_new_task(
            get_current_task, base_url=C.PERTURB_API_BASE_URL, last_task_id=None, boundary=0.0, deadline=time.time() + 15
        )
        if task is None:
            logger.error("no current task")
            return 1
        clean_b64 = download_with_retries(image_url_to_b64, task.image_url, deadline=time.time() + 15)
        record = attack_and_score(task, clean_b64, time.time() + budget)
        _write_json(out, {"mode": "current", "budget": budget, "rounds": [record]})
        return 0

    cadence = int(C.TASK_CADENCE_SECONDS)
    records: list[dict] = []
    last_task_id = None
    for i in range(rounds):
        boundary = _next_boundary(time.time(), cadence)
        if boundary - 2.0 > time.time():
            time.sleep(boundary - 2.0 - time.time())
        task = fetch_new_task(
            get_current_task,
            base_url=C.PERTURB_API_BASE_URL,
            last_task_id=last_task_id,
            boundary=boundary,
            deadline=boundary + SUBMIT_BY_SECONDS - UPLOAD_RESERVE_SECONDS - MIN_ATTACK_SECONDS,
        )
        if task is None:
            records.append({"round": i, "boundary": boundary, "task_id": None, "official": None})
            continue
        seen = time.time() - boundary
        last_task_id = task.task_id
        epoch = task_epoch(task.task_id)
        if epoch is not None and abs(time.time() - epoch) < 60:
            seen += boundary - epoch
            boundary = float(epoch)
        deadline = boundary + SUBMIT_BY_SECONDS - UPLOAD_RESERVE_SECONDS
        clean_b64 = download_with_retries(image_url_to_b64, task.image_url, deadline=deadline - MIN_ATTACK_SECONDS)
        start = time.time() - boundary
        record = attack_and_score(task, clean_b64, deadline)
        record |= {
            "round": i,
            "boundary": boundary,
            "seen_since_boundary": seen,
            "attack_start_since_boundary": start,
            "done_since_boundary": start + record["attack_seconds"],
        }
        logger.info(
            f"round {i + 1}/{rounds}: seen +{seen:.1f}s, attack +{start:.1f}s..+{record['done_since_boundary']:.1f}s "
            f"(submit-by {SUBMIT_BY_SECONDS:.0f}s, upload reserve {UPLOAD_RESERVE_SECONDS:.0f}s)"
        )
        records.append(record)
        _write_json(out, {"mode": "live_rounds", "summary": _summarize_rounds(records), "rounds": records})
    summary = _summarize_rounds(records)
    logger.info(f"dry-run summary {json.dumps(summary)}")
    _write_json(out, {"mode": "live_rounds", "summary": summary, "rounds": records})
    return 0 if summary["scored"] == rounds else 1


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
            _setup_miner_logging()  # after upstream configured bittensor logging, which mutes other loggers
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
    pre.add_argument("--rounds", type=int, default=0, help="dry run: wait for this many consecutive live tasks")
    pre.add_argument("--out", default=None, help="dry run: write per-round results as JSON")
    ours, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0], *rest]  # upstream build_config parses the remaining flags
    if ours.dry_run:
        return dry_run(ours.budget, ours.rounds, ours.out)
    return run_miner()


if __name__ == "__main__":
    raise SystemExit(main())
