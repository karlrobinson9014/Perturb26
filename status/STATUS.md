# Status

## Goal
Subnet: Perturb, NETUID 26 on finney.
Take scanning rank 1 on Perturb: mean official score above every competitor, consistently.
Competitor envelope on dev = 0.9784 mean best-per-image, 0.9858 max (`notes/competition.md`).
Ceiling 0.99303. Loop stop target `TARGET_SCORE` = 0.993 (harness/loop.env).

## Current state (2026-10-19, bootstrap)
- Product: `miner26/` v0 attack (quantized STE search + L1 + greedy prune + novelty pad), live neuron
  subclassing upstream `PerturbMiner`, `--dry-run` mode.
- Bench: `bench/run_bench.py` with the official upstream scorer under 3 numeric variants; dev manifest of 64
  real task images (`bench/manifests/dev.json`); holdout (128, test split) only on the GPU box, outside the repo.
- Tests: contract tests pass on CPU (scoring parity, alias collision, attack admissibility on TinyNet).
- v0 on the real model, CPU only, 2 dev images, 300 s budget: both flip at 1/255 (first flip at step 9),
  official 0.963 / 0.942, mean 0.953, versus the best competitor on the same images 0.981 / 0.975.
  Margin sits at the 1.0 safety floor and the nonzero count is high: room to improve.
  **Not yet run on GPU at the live 22 s budget.** No accepted best yet: the harness seeds
  `status/best.json` from HEAD on first start.
- Harness: passed independent review after three rounds (findings fixed: ledger deadlock, rename bypass,
  allowlist, scorer isolation, integrity checks, holdout handling). Accept, reject, path-violation,
  plant-a-sitecustomize and noop paths were exercised end to end on CPU with a fake claude binary.
  Residual risk: with `AGENT_RUN_PREFIX` empty, agents run as the harness user (README "Isolation").

## Recent iterations
- 2026-10-19 iter 0 (setup session): scaffold, v0 attack, harness, competitor proxy 0.9784.
