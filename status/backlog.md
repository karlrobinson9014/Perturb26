# Backlog

Top = highest priority. `[build]` items run in build mode (auto mode does them first);
`[research]` items are experiment directions for research mode (also see `notes/ideas.md`).
Format: `- [ ] [build] title: details / acceptance`. Tick with `[x]` and add the date and result.

## Build
- [x] [build] GPU baselines (2026-10-19, RTX 4090, budget 22): v0 0.9536, 64/64 success, p95 20.8 s, max 22.3 s,
  0 variant disagreements (`results/dev-v0.json`); upstream baseline 0.3371, 33/64 success
  (29 below_min_ssim, 2 above_max_delta) (`results/dev-baseline.json`).
- [x] [build] Warm-up in the live miner (2026-10-19): v1 `Attacker.__init__` captures the CUDA graphs, so the
  live miner and the bench are warm before the first task (first bench image: 296 steps).
- [x] [build] Competitor proxy on GPU with all variants (2026-10-19): 0/336 variant disagreements (max spread 2.8e-5);
  best responses stop at margin ~9.99 (median), RMSE median 0.00069 vs ours 0.00086; best competitor wins 64/64
  by mean 0.0035 (`results/competitor-variants-dev.json`, notes/competition.md).
- [x] [build] Throughput profile (2026-10-19): eager ~125 ms/step at any precision (launch-bound); CUDA-graph fp16
  18 ms (B=1), 11 ms/img (B=4), 9.4 ms/img (B=8); adopted in v1. Details in research_log 2026-10-19.
- [x] [build] Live timing probe `tools/task_probe.py` (2026-10-19, from the dev machine): task visible at
  boundary + 3.0 s median, download 0.3 s, GET /task median 2 s with spikes to 8 s and timeouts. 22 s budget
  confirmed safe. notes/timing.md. Re-run from the GPU box.
- [x] [build] Fast task fetch in the live miner (2026-10-19, `miner26/fetch.py`): 3 s per request, 0.5 s retries
  until the task deadline, errors retried instead of skipping the round, stale (previous-epoch) tasks rejected,
  image download retried. `tests/test_fetch.py` (fake HTTP server). Live: task seen at boundary + 7.3 s / + 2.2 s.
- [x] [build] Timing probe from the GPU box (2026-10-19, 10 boundaries, `results/timing-probe-gpu-2026-10-19.json`):
  task visible at boundary + 2.0 s median (1.3-2.7 s), GET /task median 0.31 s (max 6.3 s), download 0.5 s,
  0 errors. The + 7.3 s of iter 3 did not recur; ~25.6 s attack window worst case, so no budget change.
- [x] [build] Live dry runs (2026-10-19, `python -m miner26.neuron --dry-run --rounds 5 --out results/dry-run-live-2026-10-19.json`):
  5/5 consecutive live tasks scored, 5/5 success, official mean 0.97685 (min 0.96875, max 0.98328), 0 variant
  disagreements; task seen at boundary + 2.7 s median (max 3.0), attack ~25 s, done at + 28.7 s (deadline 29).
  Also fixed: bittensor muted the `miner26` logger (level CRITICAL), so the dry run and the live miner logged nothing.
- [ ] [build] [blocked: no wallet in ~/.bittensor/wallets and no perturb26-miner pm2 app on the GPU box (iter 6)] Confirm live-miner logging: after the `_setup_miner_logging` fix, check the pm2 log of perturb26-miner
  shows `Submitted task=... since_boundary=...` lines (needs the wallet / running miner; not testable in a dry run).
- [x] [build] v1 regression test on the GPU (2026-10-19, `tests/test_graph_lanes_gpu.py`): on EfficientNetV2-L and the
  first crop-path dev image, with 4 lanes (spec modes, last lane targeted) and sparse +-1 q, graph `check` and `grad`
  are bit-identical to the eager `_Lanes` (margin diff 0.0, gradient cosine 1.0, relative error 0.0); replays do not
  depend on earlier replays. Skips without CUDA, cached weights or dev data.
- [ ] [build] Early exit in the attack when the estimate cannot improve (e.g. alias images already at the
  ceiling); the attack must still respect the deadline. Unit test it on TinyNet.
- [x] [build] (file exists since c7988d7; verify it runs) `tools/leaderboard.py` (needs bittensor + NETUID): list validators (permit, stake > 10k), GET
  /leaderboard/<hotkey>, write the top-10 avgScore per validator plus a stake-weighted top list to
  status/leaderboard.json. NETUID is 26 (finney).

## Research directions (most promising first)
- [ ] [research] **Target-class discovery** (the gap to the leaders, research_log 2026-10-19): aimed at the
  competitors' class, the v1 search matches them; v1's own lanes often settle in near classes. Ideas: more
  lanes (B=8), shorter discovery then targeted lanes, candidate classes from `-l_true` / LSE lanes at several
  checkpoints, re-targeting more than once, class priors learned from the public dataset's *train* split only.
- [ ] [research] RMSE at saturated margin (2026-10-19 competitor proxy): margins already match the leaders
  (median 10.02 vs 9.99), RMSE does not (0.00086 vs 0.00069). Per image, compare our nonzero count with the best
  competitor's at the same class; try pruning after the margin saturates (drop values whose removal keeps margin >= 10).
- [ ] [research] Margin-sparsity trade-off: top competitors reach ~1-3% nonzero with margin >= 10.
- [ ] [research] Reliability: drive failures and variant disagreements to zero before chasing score.
- [ ] [research] Speed: more steps per budget (see throughput notes).
- [ ] [research] Training track (20 % of miner emission, daily winner): adversarial fine-tune of
  EfficientNetV2-L; separate project, after scanning rank 1.
