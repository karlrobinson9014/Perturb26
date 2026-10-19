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
