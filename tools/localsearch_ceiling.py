"""How much can exact batched prune/swap/flip moves add after the attack, with no time limit? (iter 217)

Runs Attacker.run at the bench budget, then ``--polish`` seconds of exact local search on the returned image:
each round takes the fp32 margin gradient at the current point, builds ``--batch`` candidates (drop the k
cheapest nonzeros by g*q, random drops from the cheap pool, sign flips of entries that oppose the margin,
drop 2k + add k best zeros), scores all of them in one fp32 forward on the validator path (PREPROCESS on the
full image) and accepts the best exact improvement (margin >= safety). Prints the est gain over time.

    python -m tools.localsearch_ceiling --limit 6 --polish 60 --out results/localsearch-ceiling-iter217.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bench.official import clean_task_b64
from bench.run_bench import load_items
from miner26.attack import AttackConfig, Attacker
from miner26.imaging import b64_to_u8
from miner26.scoring import QUANT, estimate_score, rmse_for_unit_changes, validator_true_index
from perturbnet.model import PREPROCESS, load_efficientnet_v2_l


def score_batch(model, clean, deltas, true_index, total):
    with torch.no_grad():
        o = model(PREPROCESS((clean.unsqueeze(0) + deltas) / 255.0)).float()
    true = o[:, true_index].clone()
    o[:, true_index] = -float("inf")
    margin = (o.max(1).values - true).tolist()
    nz = (deltas != 0).flatten(1).sum(1).tolist()
    px = (deltas != 0).any(dim=1).flatten(1).sum(1).tolist()
    ests = [estimate_score(linf=QUANT, rmse=rmse_for_unit_changes(n, total), margin=m, changed_pixels=p,
                           flipped=m > 0, ) for n, m, p in zip(nz, margin, px)]
    return ests, margin, nz


def margin_grad(model, clean, delta, true_index):
    dv = delta.unsqueeze(0).clone().requires_grad_(True)
    o = model(PREPROCESS((clean.unsqueeze(0) + dv) / 255.0)).float()
    true = o[0, true_index]
    others = o[0].clone()
    others[true_index] = -float("inf")
    (g,) = torch.autograd.grad(others.max() - true, dv)
    return g[0]


def candidates(delta, g, lo, hi, mask, batch, gen):
    q = delta
    nzm = (q != 0) & mask
    cost = torch.where(nzm, g * q, torch.full_like(g, float("inf"))).flatten()
    order = torch.argsort(cost)
    n_nz = int(nzm.sum())
    order = order[:n_nz]
    # zeros we can add, with the sign that raises the margin
    s = torch.sign(g)
    room = ((s > 0) & (hi >= 1)) | ((s < 0) & (lo <= -1))
    addable = (q == 0) & mask & room
    gain = torch.where(addable, g.abs(), torch.full_like(g, -1.0)).flatten()
    add_order = torch.argsort(gain, descending=True)
    # flips: nonzero entries opposing the margin, with room on the other side
    flip_ok = nzm & (g * q < 0) & (((q > 0) & (lo <= -1)) | ((q < 0) & (hi >= 1)))
    fcost = torch.where(flip_ok, g * q, torch.full_like(g, float("inf"))).flatten()
    n_flip = int(flip_ok.sum())
    forder = torch.argsort(fcost)[:n_flip]
    out, tags = [], []

    def drop(idx, base=None):
        c = (q if base is None else base).clone().flatten()
        c[idx] = 0
        return c.view_as(q)

    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    for k in ks:
        if k <= n_nz:
            out.append(drop(order[:k])); tags.append(f"drop{k}")
    for k in (16, 64, 256, 1024):
        pool = order[: min(n_nz, 4 * k)]
        if pool.numel() >= k:
            pick = pool[torch.randperm(pool.numel(), generator=gen)[:k].to(pool.device)]
            out.append(drop(pick)); tags.append(f"rdrop{k}")
    for k in (4, 16, 64, 256):
        if k <= n_flip:
            c = q.clone().flatten()
            c[forder[:k]] = -c[forder[:k]]
            out.append(c.view_as(q)); tags.append(f"flip{k}")
            # flip k, then drop the 2k cheapest non-flipped
            rest = order[~torch.isin(order, forder[:k])][: 2 * k]
            c2 = c.clone(); c2[rest] = 0
            out.append(c2.view_as(q)); tags.append(f"flip{k}drop{2*k}")
    for k in (8, 32, 128, 512):
        if 2 * k <= n_nz:
            c = q.clone().flatten()
            c[order[: 2 * k]] = 0
            a = add_order[:k]
            c[a] = s.flatten()[a]
            out.append(c.view_as(q)); tags.append(f"drop{2*k}add{k}")
    return out[:batch], tags[:batch]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--budget", type=float, default=22.0)
    ap.add_argument("--polish", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--safety", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    device = torch.device("cuda")
    model = load_efficientnet_v2_l(device)
    attacker = Attacker(model, device, AttackConfig())
    manifest = Path(__file__).resolve().parents[1] / "bench" / "manifests" / "dev.json"
    items = load_items(manifest, args.start + args.limit)[args.start:]
    gen = torch.Generator().manual_seed(0)
    rows = []
    for item, path in items:
        clean_u8 = b64_to_u8(clean_task_b64(path.read_bytes()))
        t0 = time.time()
        res = attacker.run(clean_u8, deadline=t0 + args.budget)
        clean = clean_u8.to(device).float()
        total = clean.numel()
        ti = validator_true_index(res.clean_index)
        true_index = res.clean_index if ti is None else ti
        delta = res.adv_u8.to(device).float() - clean
        lo = (-clean).clamp(min=-1.0)
        hi = (255.0 - clean).clamp(max=1.0)
        mask = attacker.influence_mask(tuple(clean.shape))
        est0, m0, nz0 = score_batch(model, clean, delta.unsqueeze(0), true_index, total)
        cur, cur_m = est0[0], m0[0]
        trace, accepted, rounds, stalls = [], {}, 0, 0
        tp = time.time()
        while time.time() - tp < args.polish:
            g = margin_grad(model, clean, delta, true_index)
            cands, tags = candidates(delta, g, lo, hi, mask, args.batch, gen)
            ests, ms, nzs = score_batch(model, clean, torch.stack(cands), true_index, total)
            rounds += 1
            ok = [i for i in range(len(cands)) if ms[i] >= args.safety and ests[i] > cur + 1e-9]
            if ok:
                i = max(ok, key=lambda i: ests[i])
                delta, cur, cur_m = cands[i], ests[i], ms[i]
                accepted[tags[i]] = accepted.get(tags[i], 0) + 1
            for T in (1, 2, 5, 10, 20, 40, 60, 120):
                if time.time() - tp >= T and not any(t[0] == T for t in trace):
                    trace.append((T, cur - est0[0]))
            stalls = 0 if ok else stalls + 1
            if stalls >= 20:  # only the random drops change between rounds; 20 misses in a row = stuck
                trace.append(("stuck_s", round(time.time() - tp, 2)))
                break
        nz_end = int((delta != 0).sum())
        row = {"key": item["key"], "attack_est": res.est_score, "attack_margin": res.margin, "nonzero0": nz0[0],
               "est_end": cur, "gain": cur - est0[0], "margin_end": cur_m, "nonzero_end": nz_end, "rounds": rounds,
               "trace": trace, "accepted": accepted}
        rows.append(row)
        print(json.dumps(row), flush=True)
    gains = [r["gain"] for r in rows]
    summary = {"n": len(rows), "mean_gain": sum(gains) / len(gains), "polish_s": args.polish, "batch": args.batch}
    print(json.dumps(summary), flush=True)
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
