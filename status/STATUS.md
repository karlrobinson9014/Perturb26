# Status

## Goal
Subnet: Perturb, NETUID 26 on finney.
Take scanning rank 1 on Perturb: mean official score above every competitor, consistently.
Competitor envelope on dev = 0.9784 mean best-per-image, 0.9858 max (`notes/competition.md`).
Ceiling 0.99303. Loop stop target `TARGET_SCORE` = 0.993 (harness/loop.env).

## Current state (2026-10-19, human-supervised session)
- Product: `miner26/` v1 attack: 4-lane batched ISTA (margin / LSE / -logit_true objectives) on CUDA graphs,
  weak lanes re-target surfaced classes at 35 % of the budget; generic eager path for non-crop geometries.
  Dev (16 images): **0.9738** (v0 0.9553), 16/16 success. Competitors' best per image on the same images 0.9771;
  live rank 1 (uid 36) averages 0.9801 (`notes/competition.md`).
- The gap is the target class (research_log 2026-10-19): aimed at the class the leaders chose, the same search
  matches them. First research priority: target-class discovery (backlog).
- Harness: operator directives / idea queue ($OPERATOR_DIR, dashboard), Consensus + SciSpace for literature
  (`notes/literature.md`), other claude.ai connectors denied, IS_SANDBOX for root containers, dashboard
  (`harness/dashboard.py`, pm2 app perturb26-dashboard), blocked build items no longer pin build mode.
- Residual risk: agents run as root (IS_SANDBOX=1), so they could read the holdout (README "Isolation").

## Recent iterations
- 2026-10-19 iter 0 (setup session): scaffold, v0 attack, harness, competitor proxy 0.9784.
- 2026-10-19 (setup session): GPU baselines, CUDA-graph v1 attack 0.9738 on 16 dev images, class-choice finding, operator dashboard.
- 2026-10-19 iter 2 (build): competitor proxy on GPU, 3 variants: 0/336 disagreements, leaders' margins ~9.99, their RMSE 0.00069 vs ours 0.00086; gap 0.0035, they win 64/64 (`results/competitor-variants-dev.json`).
- 2026-10-19 iter 3 (build): fast task fetch in the live miner (`miner26/fetch.py`, 3 s timeout / 0.5 s retries, stale-epoch guard, download retries); live: task seen at boundary + 7.3 s and + 2.2 s; dry run official 0.97051; dry-run logging fixed.
- 2026-10-19 iter 4 (build): GPU-box timing probe, 10 boundaries: task at boundary + 2.0 s median (max 2.7 s), GET /task 0.31 s median, 0 errors; 22 s budget safe (~25.6 s worst-case window), no proposal (`results/timing-probe-gpu-2026-10-19.json`).
- 2026-10-19 iter 5 (build): live dry runs, 5 consecutive tasks with live timing (`--dry-run --rounds 5`): official mean 0.97685, 5/5 success, done at boundary + 28.7 s; fixed bittensor muting all `miner26` logs (dry run and live miner) (`results/dry-run-live-2026-10-19.json`).
- 2026-10-19 iter 6 (build): live-miner logging check blocked (no wallet / miner on the box); added GPU regression test `tests/test_graph_lanes_gpu.py`: CUDA-graph lanes bit-identical to eager on EfficientNetV2-L + a dev image (margins, argmax, gradients); gates pass (smoke 3/3, mean 0.96880 at budget 8).
- 2026-10-19 iter 201 (build): early exit when no candidate can beat the found one (alias images stop after 1 step at the ceiling); `tests/test_attack_early_exit.py`; gates pass, smoke 0.96927 (budget 8).
- 2026-10-19 iter 202 (research): greedy prune after the search: 16-img 0.97300 vs 0.97365, reverted; candidates are already tight at margin 10 (research_log).
- 2026-10-19 iter 203 (research): batched fp32 checks every 8 steps (`check_every`): 302 -> 437 steps/image, dev 0.97529 vs best 0.97484, 64/64 success, 0 disagreements (`results/dev-iter203-agent.json`).
- 2026-10-20 iter 204 (research): top-3 logit target classes + extra re-target round at 0.6 (with iter 203 check_every=8): dev 0.97561 vs best 0.97484, 64/64, 0 disagreements; B=8 lanes worse; surfaced classes exhausted (`results/dev-iter204-agent.json`).
- 2026-10-20 iter 205 (research): ISTA shrink weighted toward saturated entries (`rw_power=2`, reweighted L1 itself was worse), on iter 204's code: dev 0.97628 vs best 0.97484 (+0.00144, below MIN_GAIN), 64/64, 0 disagreements, nonzero median 39.3k vs 44.2k (`results/dev-iter205-agent.json`).
- 2026-10-20 iter 206 (research): re-applied iter 203+204+205 stacked (check_every=8, top-3 re-target + round at 0.6, rw_power=2): dev 0.97622 vs best 0.97484 (+0.00138), 64/64, 0 disagreements; B=8 lanes 16-img -0.0007, B=6 tie, not kept (`results/dev-iter206-agent.json`).
- 2026-10-20 iter 207 (research): stack re-applied as default + `rw_step` option (smoothed-step shrink at 0.5, off); 16-img sweep of step profile / lr / lam all within +-0.00015 (sharp step -0.0006); full dev step s2 0.97632 vs stack 0.97622, flat (`results/dev-iter207-agent-s2.json`). Note: iter 206 discarded at 0.97618 > best+0.0005, so the running loop may still use MIN_GAIN 0.002 (pm2 restart needed?).
- 2026-10-20 iter 208 (research): momentum on the ISTA step (EMA of the normalised gradient, beta 0.5) as default; 16-img beta 0.5 +0.0005, 0.8 +0.0002, FISTA extrapolation 0.3 +0.0003; full dev 0.97649 vs best 0.97633 (+0.00017), 47/64 up, 64/64, nonzero median 37.3k vs 39.2k (`results/dev-iter208-agent.json`).
- 2026-10-20 iter 209 (research): momentum 0.5 (iter 208) + re-targeted lanes warm-started from the leader's iterate (`retarget_warm`); per-lane lr decay (`lr_final`, MDCS) 16-img -0.0001..-0.0004, off; full dev 0.97683 vs best 0.97633 (+0.00050), 45/64 up, 64/64, 0 disagreements, nonzero median 36.0k (`results/dev-iter209-agent.json`).
- 2026-10-20 iter 210 (research): re-targeted lanes inherit the leader's L1 weight (`retarget_lam`, homotopy warm start, Xiao & Zhang 2012): 16-img 1.0 -0.00015, 0.5 -0.00002; momentum 0.3 -0.0001; option kept off, defaults unchanged (`results/dev16-iter210-*.json`).
- 2026-10-20 iter 211 (research, operator directive): hard margin >= 10 (safety 10), prune floor 10.5, push to kappa 12: full dev 0.97671 vs best 0.97696 (-0.00025), 64/64, 0 disagreements, nonzero median 38.0k vs 36.0k; the hard image drops 0.9346 -> 0.9307 (no candidate reaches 10); margin was already ~10.0, so no gain to take (`results/dev-iter211-agent.json`).
- 2026-10-20 iter 212 (research, operator directive): exact batched prune/swap local search after the search (`polish_fraction`, off): it accepts ~90 % of rounds, but 16-img p0.85 -0.00036 (2/16 up) and p0.9 batch 16 -0.00017 (5/16 up). ISTA is still pruning at the deadline, so polish loses to the steps it replaces. No full dev (`results/dev16-iter212-*.json`).
- 2026-10-20 iter 213 (research, operator directive again): exact drop moves in ISTA's idle fp32 check slots (free, ~50/run) + 0.4 s check-only final rounds (`prune_slots`, `prune_final`, off): 16-img -0.00008 / -0.00003 vs accepted, 16/16, 0 disagreements; neutral, accepts get superseded by later ISTA candidates. Prune/swap around the ISTA point is closed (iters 202/212/213) (`results/dev16-iter213-*.json`).
- 2026-10-20 iter 214 (research, operator directive, third time): exact local search with 64 candidates per fp16 screen forward, top 4 in fp32 (`polish_screen`, off): 16-img -0.00003 (5/16 up), neutral; probes bound local moves at ~5e-5/image (only ~10-200 entries per image are free; fp16 forward only 1.9x cheaper than fp32); first fp16 batch-64 call warmup overran image 1 (22.28 s) (`results/dev16-iter214-screen64.json`).
- 2026-10-20 iter 215 (research, operator directive, fourth time): exact-scored rounding-threshold drop/add/swap moves on every lane's d in the last 0.45 s (`sweep_final`, off): probe +1e-5..5e-5/image, 16-img 0.97628 vs 0.97630 (-0.00002, 9/16 up), 16/16, 0 disagreements; neutral, prune/swap directive should be retired (`results/dev16-iter215-sweep045.json`).
- 2026-10-20 iter 216 (research, operator directive, fifth time): late support freeze (Lin 2023) + exact batched drops of the leader in idle check slots, fed back into d and the frozen mask (`freeze_at`/`freeze_tau`/`freeze_drop`, off): 16-img freeze 0.75 -0.00040 (1/16 up), +drops -0.00063, freeze 0.9 tau 0.25 +drops +0.00007 (9/16, noise); 16/16, 0 disagreements. Freezing removes the late ISTA swaps that keep shrinking the support. No full dev (`results/dev16-iter216-*.json`).
- 2026-10-20 iter 217 (research, operator directive, sixth time): time-unlimited ceiling of exact batched prune/swap/flip after the 22 s attack (`tools/localsearch_ceiling.py`): 8 dev images, mean +1.6e-4/image, stalls after 8-15 s, sign flips never accepted, ~90 % of the gain from 2 images with margin overshoot (10.09, 10.2). Move set exhausted at any time budget; retire the directive (`results/localsearch-ceiling-iter217.json`).
- 2026-10-20 iter 218 (research, operator directive, seventh time): exact prune/swap only on overshoot images (`clip_time` reserve when the leader's margin > 10.05, adaptive drop-k ladder + drop-2a/add-a swaps, 8 candidates per round in batched fp32 checks; off): per-image +1e-5..+5.4e-4 on the ~25 % of images that overshoot, 16-img 0.6 s -0.00015, 1.0 s -0.000004 (6/16 up), 16/16, 0 disagreements; neutral, no full dev (`results/dev16-iter218-clip10.json`).
- 2026-10-20 iter 219 (research, operator directive, eighth time): sign-opposed drops inside ISTA (`sign_drop`, off; zero support entries with g*q < 0, |g| >= t x mean): 16-img t=4 -0.00112, 8 -0.00055, 16 -0.00019 (4-5/16 up), 16/16, 0 disagreements; monotone loss, support grows 2-3x on 2 images. Prune/swap closed in every placement (`results/dev16-iter219-sd*.json`).
