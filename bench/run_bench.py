"""PROTECTED: benchmark an attack on a manifest of real task images.

    python -m bench.run_bench --manifest bench/manifests/dev.json --out results/dev-latest.json
    python -m bench.run_bench --limit 8 --budget 8                 # quick check
    python -m bench.run_bench --attack baseline                    # upstream reference miner

Two phases, two processes:
1. attack (this process): each clean task image gets ``--budget`` seconds of wall clock,
   the time a live miner has; the adversarial PNGs go to a temp directory.
2. score (``bench/score.py``, fresh process, imports nothing from miner26/): the
   upstream verifier under several numeric variants; the official per-image score is
   the minimum over variants.

The headline metric is ``mean_official``. The last stdout line is ``BENCH_RESULT {json}``.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from bench import _paths  # noqa: F401
from bench.official import VARIANTS, clean_task_b64

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUDGET = 22.0  # seconds; see notes/timing.md


def load_items(manifest_path: Path, limit: int | None) -> list[tuple[dict, Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_dir = (manifest_path.parent / manifest["data_dir"]).resolve()
    items = []
    for item in manifest["items"][: limit or None]:
        path = data_dir / item["file"]
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run python -m bench.fetch_data --manifest {manifest_path}")
        items.append((item, path))
    return items


def baseline_attack(model: torch.nn.Module, device: torch.device):
    """The upstream reference miner's ``_attack_image``, run unchanged."""
    from neurons.miner import PerturbMiner
    from perturbnet import constants as C

    stub = SimpleNamespace(model=model, device=device, config=SimpleNamespace(perturb=SimpleNamespace(**C.VALIDATOR_CONFIG)))

    def run(task_id: str, clean_b64: str) -> str:
        adv_b64, _ = PerturbMiner._attack_image(stub, task_id=task_id, clean_image_b64=clean_b64)
        return adv_b64

    return run


def attack_phase(args, items, adv_dir: Path) -> tuple[list[dict], dict]:
    from miner26.attack import AttackConfig, Attacker
    from miner26.imaging import b64_to_u8, u8_to_b64
    from perturbnet.model import load_efficientnet_v2_l

    device = torch.device(args.device)
    model = load_efficientnet_v2_l(device)
    cfg = AttackConfig(**json.loads(args.attack_config)) if args.attack_config else AttackConfig()
    attacker = Attacker(model, device, cfg) if args.attack == "v0" else None
    baseline = baseline_attack(model, device) if args.attack == "baseline" else None
    records = []
    for index, (item, path) in enumerate(items, start=1):
        key = item["key"]
        clean_b64 = clean_task_b64(path.read_bytes())
        t0 = time.time()
        info: dict = {}
        if attacker is not None:
            result = attacker.run(b64_to_u8(clean_b64), deadline=t0 + args.budget)
            adv_b64 = u8_to_b64(result.adv_u8)
            info = {"est_score": result.est_score, "est_margin": result.margin, "nonzero": result.nonzero,
                    "changed_pixels": result.changed_pixels, "steps": result.steps,
                    "alias_target": result.info.get("alias_target"), "first_flip_step": result.info.get("first_flip_step")}
        else:
            adv_b64 = baseline(f"bench-{key}", clean_b64)
        elapsed = time.time() - t0
        (adv_dir / f"{key}.png").write_bytes(base64.b64decode(adv_b64))
        records.append({"key": key, "elapsed": round(elapsed, 3), **info})
        print(f"[attack {index}/{len(items)}] {key} elapsed={elapsed:.1f}s " + " ".join(f"{k}={v}" for k, v in info.items()),
              flush=True)
    meta = {"attack": args.attack, "budget": args.budget, "attack_config": vars(cfg) if attacker is not None else None}
    del model, attacker, baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "bench" / "manifests" / "dev.json")
    parser.add_argument("--attack", choices=["v0", "baseline"], default="v0")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--attack-config", default="", help='JSON overrides for AttackConfig, e.g. \'{"kappa": 6}\'')
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--save-adv", type=Path, default=None, help="directory to keep the adversarial PNGs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    items = load_items(args.manifest, args.limit)
    work = Path(tempfile.mkdtemp(prefix="perturb26-bench-"))
    adv_dir = work / "adv"
    adv_dir.mkdir()
    attack_started = time.time()
    records, meta = attack_phase(args, items, adv_dir)
    meta["attack_wall_seconds"] = round(time.time() - attack_started, 1)
    records_path = work / "records.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONSTARTUP",)}
    env["PYTHONPATH"] = str(REPO_ROOT)
    command = [sys.executable, "-m", "bench.score", "--manifest", str(args.manifest.resolve()), "--adv-dir", str(adv_dir),
               "--records", str(records_path), "--variants", args.variants, "--device", args.device,
               "--meta", json.dumps(meta)]
    if args.out:
        command += ["--out", str(args.out.resolve())]
    rc = subprocess.call(command, cwd=REPO_ROOT, env=env)
    if args.save_adv:
        shutil.copytree(adv_dir, args.save_adv, dirs_exist_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
