"""Where do re-targeted lanes end up? Runs the attack and reads its per-lane info.

A re-targeted lane is aimed at a surfaced class (``lanes.j``), but it is kept on the
untargeted validator margin, so it may settle in another class. With ``retarget_warm``
it starts from the leading lane's iterate, which already holds margin ~10 in the
leader's class, and may simply stay there (a duplicate of the leader).

    python -m tools.lane_probe --limit 16 --out results/lane-probe-warm.json
    python -m tools.lane_probe --limit 16 --attack-config '{"retarget_warm": false}' --out results/lane-probe-cold.json

No official scoring: ``est_score`` is the attacker's own fp32 validator-path estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.official import clean_task_b64  # noqa: E402
from bench.run_bench import DEFAULT_BUDGET, load_items  # noqa: E402
from miner26.attack import AttackConfig, Attacker  # noqa: E402
from miner26.imaging import b64_to_u8  # noqa: E402
from perturbnet.model import load_efficientnet_v2_l  # noqa: E402


def classify(info: dict) -> tuple[list[dict], int | None]:
    """One row per re-targeted lane (its last re-target), plus the winning lane index."""
    lanes = info["lanes"]
    last: dict[int, tuple[int, dict]] = {}
    for rnd, sw in enumerate(info["switch"]):
        for b, target in zip(sw["lanes"], sw["targets"]):
            last[b] = (rnd, {"target": target, "leader_class": sw.get("leader_class")})
    rows = []
    for b, (rnd, sw) in sorted(last.items()):
        cls = lanes[b]["class"]
        if cls is None:
            outcome = "none"
        elif cls == sw["target"]:
            outcome = "target"
        elif cls == sw["leader_class"]:
            outcome = "leader"
        else:
            outcome = "other"
        rows.append({"lane": b, "round": rnd, "target": sw["target"], "leader_class": sw["leader_class"],
                     "class": cls, "est": lanes[b]["est"], "outcome": outcome})
    scored = [b for b, lane in enumerate(lanes) if lane["class"] is not None]
    winner = max(scored, key=lambda b: lanes[b]["est"]) if scored else None
    return rows, winner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "bench" / "manifests" / "dev.json")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    parser.add_argument("--attack-config", default="", help='JSON overrides for AttackConfig')
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_efficientnet_v2_l(device)
    cfg = AttackConfig(**json.loads(args.attack_config)) if args.attack_config else AttackConfig()
    attacker = Attacker(model, device, cfg)
    images = []
    for index, (item, path) in enumerate(load_items(args.manifest, args.limit), start=1):
        clean = b64_to_u8(clean_task_b64(path.read_bytes()))
        t0 = time.time()
        result = attacker.run(clean, deadline=t0 + args.budget)
        rows, winner = classify(result.info)
        win_retargeted = winner in {r["lane"] for r in rows}
        images.append({"key": item["key"], "est_score": result.est_score, "adv_class": result.info["adv_class"],
                       "winner": winner, "winner_retargeted": win_retargeted, "retargeted": rows,
                       "lanes": result.info["lanes"], "early_exit": result.info["early_exit"]})
        print(f"[{index}] {item['key']} est={result.est_score:.5f} winner={winner} retargeted_win={win_retargeted} "
              + " ".join(f"L{r['lane']}:{r['outcome']}" for r in rows), flush=True)

    outcomes = Counter(r["outcome"] for img in images for r in img["retargeted"])
    summary = {
        "n": len(images),
        "mean_est": sum(img["est_score"] for img in images) / max(1, len(images)),
        "retargeted_lanes": sum(outcomes.values()),
        "outcomes": dict(outcomes),
        "winner_retargeted": sum(img["winner_retargeted"] for img in images),
        "config": vars(cfg),
    }
    print("LANE_PROBE " + json.dumps({k: v for k, v in summary.items() if k != "config"}), flush=True)
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "images": images}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
