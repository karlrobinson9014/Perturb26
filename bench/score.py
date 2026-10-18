"""PROTECTED: authoritative scoring of a directory of adversarial images.

Runs in its own process and imports nothing from miner26/, so attack code cannot
patch the verifier or the clean images it is compared against. ``run_bench``
calls it after the attack phase; the harness reads the summary it writes.

    python -m bench.score --manifest M --adv-dir D --records R.json --out OUT.json [--variants ...]
"""
from __future__ import annotations

import argparse
import base64
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch

from bench import _paths  # noqa: F401
from bench.official import VARIANTS, OfficialScorer, clean_task_b64
from perturbnet.model import load_efficientnet_v2_l

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ""


def main() -> int:
    started = time.time()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adv-dir", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True, help="attack-phase records (timing, attack info)")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--meta", default="{}", help="JSON merged into the summary (attack name, budget, config)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    data_dir = (args.manifest.parent / manifest["data_dir"]).resolve()
    attack_records = {r["key"]: r for r in json.loads(args.records.read_text(encoding="utf-8"))}
    device = torch.device(args.device)
    model = load_efficientnet_v2_l(device)
    scorer = OfficialScorer(model, device, variants=tuple(v for v in args.variants.split(",") if v))

    records = []
    for item in manifest["items"]:
        key = item["key"]
        if key not in attack_records:
            continue  # --limit on the attack phase
        clean_b64 = clean_task_b64((data_dir / item["file"]).read_bytes())
        adv_path = args.adv_dir / f"{key}.png"
        if adv_path.exists():
            scored = scorer.score(f"bench-{key}", clean_b64, base64.b64encode(adv_path.read_bytes()).decode("utf-8"))
            first = scored["variants"][scorer.variants[0]]
            row = {"official": scored["official"], "max_variant": scored["max_variant"],
                   "variant_disagreement": scored["variant_disagreement"], "reasons": scored["reasons"],
                   "margin": first["margin"], "norm": first["norm"], "rmse": first["rmse"]}
        else:
            row = {"official": 0.0, "max_variant": 0.0, "variant_disagreement": False, "reasons": ["no_response"],
                   "margin": 0.0, "norm": 0.0, "rmse": 0.0}
        record = {"key": key, "image_id": item.get("image_id"), **row,
                  **{k: v for k, v in attack_records[key].items() if k not in row and k != "key"}}
        records.append(record)
        print(f"[score {len(records)}/{len(attack_records)}] {key} official={record['official']:.5f} "
              f"reasons={record['reasons']} elapsed={record.get('elapsed', 0):.1f}s", flush=True)

    officials = [r["official"] for r in records]
    times = sorted(float(r.get("elapsed", 0.0)) for r in records)
    successes = [o for o in officials if o > 0]
    summary = {
        **json.loads(args.meta),  # first, so caller-supplied meta can never override computed fields
        "manifest": str(args.manifest), "n": len(records),
        "mean_official": statistics.fmean(officials) if officials else 0.0,
        "median_official": statistics.median(officials) if officials else 0.0,
        "success_rate": len(successes) / max(1, len(officials)),
        "mean_official_on_success": statistics.fmean(successes) if successes else 0.0,
        "variant_disagreements": sum(r["variant_disagreement"] for r in records),
        "reasons": dict(Counter(reason for r in records for reason in r["reasons"])),
        "time_mean": statistics.fmean(times) if times else 0.0,
        "time_p95": times[min(len(times) - 1, int(0.95 * len(times)))] if times else 0.0,
        "time_max": times[-1] if times else 0.0,
        "variants": list(scorer.variants),
        "git_sha": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain", "--", "miner26", "bench")),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu",
        "torch": torch.__version__,
        "score_wall_seconds": round(time.time() - started, 1),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "records": records}, indent=1) + "\n", encoding="utf-8")
    print("BENCH_RESULT " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
