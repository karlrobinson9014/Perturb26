"""Score the network's own verified adversarial images with the official verifier.

The dataset stores, for each task image, the adversarial versions other miners
submitted. Their official scores show where the competition is; the per-image
maximum approximates the current best miner on that image.

    python -m bench.fetch_data --manifest bench/manifests/dev.json --with-adv
    python -m tools.competitor_scores --manifest bench/manifests/dev.json --out results/competitors-dev.json
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.official import OfficialScorer  # noqa: E402
from miner26.imaging import b64_to_u8, task_b64_from_bytes  # noqa: E402
from perturbnet.model import load_efficientnet_v2_l  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "bench" / "manifests" / "dev.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "competitors-dev.json")
    parser.add_argument("--variants", default="default")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    data_dir = (args.manifest.parent / manifest["data_dir"]).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_efficientnet_v2_l(device)
    scorer = OfficialScorer(model, device, variants=tuple(args.variants.split(",")))

    rows = []
    for index, item in enumerate(manifest["items"][: args.limit or None], start=1):
        if not item.get("adv_files"):
            continue
        clean_b64 = task_b64_from_bytes((data_dir / item["file"]).read_bytes())
        clean = b64_to_u8(clean_b64).int()
        scores = []
        for name in item["adv_files"]:
            adv_b64 = task_b64_from_bytes((data_dir / name).read_bytes())
            adv = b64_to_u8(adv_b64).int()
            result = scorer.score(item["key"], clean_b64, adv_b64)
            v = result["variants"][scorer.variants[0]]
            same_shape = adv.shape == clean.shape
            scores.append({
                "file": name, "official": result["official"], "reason": v["reason"], "norm": v["norm"],
                "rmse": v["rmse"], "margin": v["margin"],
                "linf_levels": int((adv - clean).abs().max()) if same_shape else None,
                "nonzero_fraction": float(((adv - clean) != 0).float().mean()) if same_shape else None,
            })
        best = max(scores, key=lambda s: s["official"])
        rows.append({"key": item["key"], "best": best["official"], "scores": scores})
        print(f"[{index}] {item['key']} n={len(scores)} best={best['official']:.5f} "
              f"(levels={best['linf_levels']} nz={best['nonzero_fraction']}) all={[round(s['official'], 4) for s in scores]}",
              flush=True)

    all_scores = [s["official"] for r in rows for s in r["scores"]]
    bests = [r["best"] for r in rows]
    summary = {
        "images": len(rows), "adversarials": len(all_scores),
        "mean_best_per_image": statistics.fmean(bests) if bests else 0.0,
        "median_best_per_image": statistics.median(bests) if bests else 0.0,
        "max_any": max(all_scores) if all_scores else 0.0,
        "mean_all": statistics.fmean(all_scores) if all_scores else 0.0,
        "note": "dataset rows are the network's verified submissions (score>0 at submission time); "
                "mean_best_per_image is an optimistic proxy for the top miner, which must win every image",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    print("COMPETITOR_RESULT " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
