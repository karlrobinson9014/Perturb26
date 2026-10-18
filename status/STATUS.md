# Status

## Goal
Take scanning rank 1 on Perturb: mean official score above every competitor, consistently.
Competitor envelope on dev = 0.9784 mean best-per-image, 0.9858 max (`notes/competition.md`).
Ceiling 0.99303. Loop stop target `TARGET_SCORE` = 0.993 (harness/loop.env).

## Current state (2026-10-19, bootstrap)
- Product: `miner26/` v0 attack (quantized STE search + L1 + greedy prune + novelty pad), live neuron
  subclassing upstream `PerturbMiner`, `--dry-run` mode.
- Bench: `bench/run_bench.py` with the official upstream scorer under 3 numeric variants; dev manifest of 64
  real task images (`bench/manifests/dev.json`); holdout (128, test split) only on the GPU box, outside the repo.
- Tests: contract tests pass on CPU (scoring parity, alias collision, attack admissibility on TinyNet).
- **Not yet run on the real model on GPU.** No accepted best yet: the harness seeds `status/best.json`
  from HEAD on first start.

## Recent iterations
- 2026-10-19 iter 0 (setup session): scaffold, v0 attack, harness, competitor proxy 0.9784.
