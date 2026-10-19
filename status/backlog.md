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
- [ ] [build] Fast task fetch in the live miner: override upstream's fetch with a ~3 s per-request timeout and
  0.5 s retries (upstream uses 10 s / 2 s; one timeout can cost the round). Unit test with a fake server.
- [ ] [build] Live dry runs: `python -m miner26.neuron --dry-run` on 5 consecutive live tasks; record official
  scores and wall-clock since boundary in research_log.
- [ ] [build] v1 regression test on the GPU: graph-captured lanes give the same margins and gradients as the
  eager `_Lanes` on a real dev image (skip when CUDA is unavailable).
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
