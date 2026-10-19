"""Who is winning right now: the network's consensus ranking, rebuilt from public data.

Reads the subnet metagraph, fetches every weight-setting validator's leaderboard
report (GET /leaderboard/<hotkey>, public), and recomputes the stake-weighted
consensus rank exactly as validators do before setting weights (upstream
``perturbnet.emissions``). Read-only: no wallet, no transactions.

    python -m tools.leaderboard --netuid 26 --out results/leaderboard.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import miner26  # noqa: E402,F401  (upstream on sys.path)
import bittensor as bt  # noqa: E402
from perturbnet import constants as C  # noqa: E402
from perturbnet.api_client import get_leaderboard_avg_scores  # noqa: E402
from perturbnet.emissions import ranked_emission_shares, stake_weighted_ranks  # noqa: E402
from perturbnet.metagraph_utils import is_validator_neuron, miner_incentive, neuron_stake  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--netuid", type=int, default=26)
    parser.add_argument("--network", default="finney")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "leaderboard.json")
    args = parser.parse_args()

    subtensor = bt.Subtensor(network=args.network)
    metagraph = subtensor.metagraph(netuid=args.netuid)
    n = int(metagraph.n)
    validators = [(uid, str(metagraph.hotkeys[uid]), neuron_stake(metagraph, uid))
                  for uid in range(n) if is_validator_neuron(metagraph, uid)]
    print(f"netuid={args.netuid} block={int(subtensor.get_current_block())} neurons={n} validators={len(validators)}")

    reports, per_validator = [], []
    for uid, hotkey, stake in sorted(validators, key=lambda v: -v[2]):
        try:
            scores = get_leaderboard_avg_scores(base_url=C.PERTURB_API_BASE_URL, validator_hotkey=hotkey,
                                                timeout_seconds=C.PERTURB_API_TIMEOUT_SECONDS)
        except Exception as exc:
            print(f"  validator uid={uid} stake={stake:,.0f}: no report ({exc})")
            continue
        top = sorted(((u, s) for u, s in scores.items() if s > 0), key=lambda x: -x[1])[: args.top]
        per_validator.append({"uid": uid, "hotkey": hotkey, "stake": stake, "miners_reported": len(scores),
                              "top": [{"uid": u, "avg_score": round(s, 5)} for u, s in top]})
        reports.append((stake, {u: s for u, s in scores.items() if u != C.BURN_UID}))
        print(f"  validator uid={uid} stake={stake:,.0f}: top " +
              ", ".join(f"uid{u}={s:.4f}" for u, s in top[:5]))

    ranks = stake_weighted_ranks(reports)
    ordered = sorted(ranks.items(), key=lambda item: (item[1], item[0]))
    shares = ranked_emission_shares([uid for uid, _ in ordered])
    consensus = []
    for position, (uid, rank) in enumerate(ordered[: args.top], start=1):
        avgs = [scores.get(uid, 0.0) for _, scores in reports]
        consensus.append({
            "position": position, "uid": uid, "hotkey": str(metagraph.hotkeys[uid]),
            "consensus_rank": round(rank, 3), "mean_avg_score": round(statistics.fmean(avgs), 5) if avgs else 0.0,
            "max_avg_score": round(max(avgs), 5) if avgs else 0.0, "scanning_share": shares.get(uid, 0.0),
            "incentive": round(miner_incentive(metagraph, uid), 5),
        })
    result = {"netuid": args.netuid, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "validators_reporting": len(reports), "consensus_top": consensus, "validators": per_validator}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print("consensus top:")
    for row in consensus:
        print(f"  #{row['position']} uid{row['uid']} rank={row['consensus_rank']} mean_avg={row['mean_avg_score']:.4f} "
              f"max_avg={row['max_avg_score']:.4f} share={row['scanning_share']:.2f} incentive={row['incentive']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
