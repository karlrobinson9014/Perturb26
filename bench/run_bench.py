"""Benchmark an attack on a manifest of real task images, scored by the official verifier.

    python -m bench.run_bench --manifest bench/manifests/dev.json --out results/dev-latest.json
    python -m bench.run_bench --manifest bench/manifests/dev.json --limit 8 --budget 8   # quick check
    python -m bench.run_bench --manifest bench/manifests/dev.json --attack baseline     # upstream miner

The headline metric is ``mean_official``: the mean over images of the minimum
official score across numeric variants (bench/official.py), with the attack
held to ``--budget`` seconds of wall clock per image, the time a live miner
has. The final stdout line is ``BENCH_RESULT {json summary}``.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from bench.official import VARIANTS, OfficialScorer
from miner26 import _upstream  # noqa: F401
from miner26.attack import AttackConfig, Attacker
from miner26.imaging import b64_to_u8, task_b64_from_bytes, u8_to_b64
from perturbnet import constants as C
from perturbnet.model import load_efficientnet_v2_l

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUDGET = 22.0  # seconds; see notes/timing.md


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--", "miner26", "bench"], cwd=REPO_ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return False


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
    from miner26._upstream import ensure_bittensor_importable

    ensure_bittensor_importable()
    from neurons.miner import PerturbMiner

    stub = SimpleNamespace(model=model, device=device, config=SimpleNamespace(perturb=SimpleNamespace(**C.VALIDATOR_CONFIG)))

    def run(task_id: str, clean_b64: str) -> str:
        adv_b64, _ = PerturbMiner._attack_image(stub, task_id=task_id, clean_image_b64=clean_b64)
        return adv_b64

    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "bench" / "manifests" / "dev.json")
    parser.add_argument("--attack", choices=["v0", "baseline"], default="v0")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--attack-config", default="", help='JSON overrides for AttackConfig, e.g. \'{"kappa": 6}\'')
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--save-adv", type=Path, default=None, help="directory to write adversarial PNGs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_efficientnet_v2_l(device)
    scorer = OfficialScorer(model, device, variants=tuple(v for v in args.variants.split(",") if v))
    cfg = AttackConfig(**json.loads(args.attack_config)) if args.attack_config else AttackConfig()
    attacker = Attacker(model, device, cfg) if args.attack == "v0" else None
    baseline = baseline_attack(model, device) if args.attack == "baseline" else None
    items = load_items(args.manifest, args.limit)
    if args.save_adv:
        args.save_adv.mkdir(parents=True, exist_ok=True)

    records = []
    started = time.time()
    for index, (item, path) in enumerate(items, start=1):
        task_id = f"bench-{item['offset']:06d}"
        clean_b64 = task_b64_from_bytes(path.read_bytes())
        t0 = time.time()
        info: dict = {}
        if attacker is not None:
            result = attacker.run(b64_to_u8(clean_b64), deadline=t0 + args.budget)
            adv_b64 = u8_to_b64(result.adv_u8)
            info = {"est_score": result.est_score, "est_margin": result.margin, "nonzero": result.nonzero,
                    "changed_pixels": result.changed_pixels, "steps": result.steps,
                    "alias_target": result.info.get("alias_target"), "first_flip_step": result.info.get("first_flip_step")}
        else:
            adv_b64 = baseline(task_id, clean_b64)
        elapsed = time.time() - t0
        scored = scorer.score(task_id, clean_b64, adv_b64)
        if args.save_adv:
            import base64

            (args.save_adv / f"{item['offset']:06d}.png").write_bytes(base64.b64decode(adv_b64))
        record = {"offset": item["offset"], "image_id": item.get("image_id"), "elapsed": round(elapsed, 3),
                  "official": scored["official"], "max_variant": scored["max_variant"],
                  "variant_disagreement": scored["variant_disagreement"], "reasons": scored["reasons"],
                  "margin": scored["variants"][scorer.variants[0]]["margin"],
                  "norm": scored["variants"][scorer.variants[0]]["norm"],
                  "rmse": scored["variants"][scorer.variants[0]]["rmse"], **info}
        records.append(record)
        print(f"[{index}/{len(items)}] {task_id} official={record['official']:.5f} reasons={record['reasons']} "
              f"elapsed={elapsed:.1f}s " + " ".join(f"{k}={v}" for k, v in info.items()), flush=True)

    officials = [r["official"] for r in records]
    times = sorted(r["elapsed"] for r in records)
    summary = {
        "manifest": str(args.manifest), "attack": args.attack, "budget": args.budget, "n": len(records),
        "mean_official": statistics.fmean(officials) if officials else 0.0,
        "median_official": statistics.median(officials) if officials else 0.0,
        "success_rate": sum(o > 0 for o in officials) / max(1, len(officials)),
        "mean_official_on_success": statistics.fmean([o for o in officials if o > 0]) if any(o > 0 for o in officials) else 0.0,
        "variant_disagreements": sum(r["variant_disagreement"] for r in records),
        "reasons": dict(Counter(reason for r in records for reason in r["reasons"])),
        "time_mean": statistics.fmean(times) if times else 0.0,
        "time_p95": times[min(len(times) - 1, int(0.95 * len(times)))] if times else 0.0,
        "time_max": times[-1] if times else 0.0,
        "attack_config": vars(cfg) if attacker is not None else None,
        "variants": list(scorer.variants), "git_sha": _git_sha(), "git_dirty": _git_dirty(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu",
        "torch": torch.__version__, "wall_seconds": round(time.time() - started, 1),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "records": records}, indent=1) + "\n", encoding="utf-8")
    print("BENCH_RESULT " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
